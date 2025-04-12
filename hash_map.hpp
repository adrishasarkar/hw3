#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <list>
#include <vector>
#include <cmath>
#include <iostream> // Add for debugging

#define UPCXX_SEGMENT_MB 120

struct HashMap {

    // Node structure for linked list chaining
    struct Node {
        kmer_pair data;
        Node* next;
        
        Node(const kmer_pair& kmer) : data(kmer), next(nullptr) {}
    };
    
    // Vector of linked lists (buckets)
    std::vector<Node*> buckets;
    
    size_t my_size;    // Local portion size
    size_t total_size; // Total size across all ranks
    
    upcxx::dist_object<HashMap*> dobj;
    
    // Global pointers for receiving kmers from other ranks
    std::vector<upcxx::global_ptr<kmer_pair>> recv_ptrs;
    
    // Global pointers for sending kmers to other ranks
    std::vector<upcxx::global_ptr<kmer_pair>> send_ptrs;
    
    // Counts of kmers to be received from each rank
    std::vector<size_t> recv_counts;
    
    // Counts of kmers to be sent to each rank
    std::vector<size_t> send_counts;
    
    HashMap(size_t size);
    ~HashMap();
    
    // Size functions
    size_t size() const noexcept;
    size_t local_size() const noexcept;
    
    // Main API functions
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);
    
    // Local operations (for this rank's portion)
    bool local_insert(const kmer_pair& kmer);
    bool local_find(const pkmer_t& key_kmer, kmer_pair& val_kmer);
    
    int get_target_rank(const pkmer_t& key_kmer) const;
    uint64_t get_local_slot(uint64_t hash_val) const;
    
    // Helper functions for memory management
    void clear_buckets();

    // Batch functions
    void process_kmers(const std::vector<kmer_pair>& kmers);
    void batch_find_kmers(const std::vector<pkmer_t>& key_kmers, std::vector<kmer_pair> & results);
};

HashMap::HashMap(size_t size) : dobj(this) {
    // Calculate local portion size
    int rank_n = upcxx::rank_n();
    int rank_me = upcxx::rank_me();
    
    // Distribute size evenly, with remainder going to first ranks
    size_t base_size = size / rank_n;
    size_t remainder = size % rank_n;
    
    my_size = base_size + (rank_me < remainder ? 1 : 0);
    total_size = size;
    
    // Initialize buckets with nullptr
    buckets.resize(my_size, nullptr);
    
    // Initialize receive pointers vector
    recv_ptrs.resize(rank_n);
    
    // Initialize send pointers vector
    send_ptrs.resize(rank_n);
    
    // Initialize receive counts vector
    recv_counts.resize(rank_n, 0);
    
    // Initialize send counts vector
    send_counts.resize(rank_n, 0);
    
    // Ensure all ranks are initialized before proceeding
    upcxx::barrier();
}

HashMap::~HashMap() {
    clear_buckets();
    
    // Clean up any allocated memory for receiving
    int rank_n = upcxx::rank_n();
    for (int i = 0; i < rank_n; ++i) {
        if (send_ptrs[i] != nullptr) {
            upcxx::delete_array(send_ptrs[i]);
        }
    }
}

void HashMap::clear_buckets() {
    for (size_t i = 0; i < my_size; ++i) {
        Node* current = buckets[i];
        while (current != nullptr) {
            Node* next = current->next;
            delete current;
            current = next;
        }
        buckets[i] = nullptr;
    }
}

// Get total hash table size
size_t HashMap::size() const noexcept {
    return total_size;
}

// Get local portion size
size_t HashMap::local_size() const noexcept {
    return my_size;
}

// Determine which rank owns this key
int HashMap::get_target_rank(const pkmer_t& key_kmer) const {
    return key_kmer.hash() % upcxx::rank_n();
}

// Calculate local slot from hash
uint64_t HashMap::get_local_slot(uint64_t hash_val) const {
    return hash_val % my_size;
}

// Insert function
bool HashMap::insert(const kmer_pair& kmer) {
    int target_rank = get_target_rank(kmer.kmer);
    
    if (target_rank == upcxx::rank_me()) {
        // Local insert
        return local_insert(kmer);
    } else {
        // Remote insert using RPC
        return upcxx::rpc(target_rank,
            [](const kmer_pair& kmer, upcxx::dist_object<HashMap*>& dobj) {
                return (*dobj)->local_insert(kmer);
            }, kmer, dobj).wait();
    }
}

// Process kmers using the new approach
void HashMap::process_kmers(const std::vector<kmer_pair>& kmers) {
    int rank_n = upcxx::rank_n();
    int rank_me = upcxx::rank_me();
    
    // Organize kmers by target rank
    std::vector<std::vector<kmer_pair>> kmers_by_rank(rank_n);
    
    for (const auto& kmer : kmers) {
        int target_rank = get_target_rank(kmer.kmer);
        kmers_by_rank[target_rank].push_back(kmer);
    }

    // Process local kmers first for immediate progress
    for (const auto& kmer : kmers_by_rank[rank_me]) {
        local_insert(kmer);
    }
    
    // Calculate counts for sending and receiving
    for (int i = 0; i < rank_n; ++i) {
        send_counts[i] = kmers_by_rank[i].size();
    }

    size_t max_num_kmers_send = 0;
    for (int i = 0; i < rank_n; ++i) {
        if (i != rank_me) {
            max_num_kmers_send = std::max(max_num_kmers_send, send_counts[i]);
        }
    }

    upcxx::barrier();

    size_t global_max_num_kmers_send = upcxx::reduce_all(max_num_kmers_send, upcxx::op_fast_max).wait();
    
    size_t seg_size = static_cast<size_t>(UPCXX_SEGMENT_MB) * 1024 * 1024;
    size_t seg_num_kmers_per_rank = seg_size / ((rank_n - 1) * sizeof(kmer_pair)); // in Bytes
    size_t seg_size_per_rank = seg_num_kmers_per_rank * sizeof(kmer_pair); // in Bytes

    size_t num_chunks = std::ceil(static_cast<double>(global_max_num_kmers_send) / seg_num_kmers_per_rank);
    
    // All-to-all exchange of counts
    for (int i = 0; i < rank_n; ++i) {
        if (i != rank_me) {
            upcxx::rpc(i, 
                [](size_t count, int sender_rank, upcxx::dist_object<HashMap*>& dobj) {
                    (*dobj)->recv_counts[sender_rank] = count;
                }, send_counts[i], rank_me, dobj).wait();
        } else {
            recv_counts[i] = send_counts[i];
        }
    }
    
    // Ensure all ranks have received their counts
    upcxx::barrier();

    // Allocate memory for sending kmers
    for (int i = 0; i < rank_n; ++i) {
        if (i != rank_me && send_counts[i] > 0) {
            // Clean up any previous allocation
            if (send_ptrs[i] != nullptr) {
                upcxx::delete_array(send_ptrs[i]);
                send_ptrs[i] = nullptr;
            }
            
            size_t array_num_elems = std::max(send_counts[i], seg_num_kmers_per_rank);

            // Allocate memory for receiving
            send_ptrs[i] = upcxx::new_array<kmer_pair>(array_num_elems);
            
            // Send the pointer to the receiver rank
            if (send_ptrs[i] != nullptr) {
                upcxx::rpc(i, 
                    [](upcxx::global_ptr<kmer_pair> ptr, int sender_rank, upcxx::dist_object<HashMap*>& dobj) {
                        (*dobj)->recv_ptrs[sender_rank] = ptr;
                    }, send_ptrs[i], rank_me, dobj).wait();
            }
        }
    }
    
    // Ensure all ranks have received their pointers
    upcxx::barrier();

    std::vector<size_t> sent_counters(rank_n, 0);
    std::vector<size_t> rcvd_counters(rank_n, 0);

    for (int send_iter = 0; send_iter < num_chunks; ++send_iter) {
        for (int i = 0; i < rank_n; ++i) {
            size_t to_send = std::min(seg_num_kmers_per_rank, send_counts[i] - sent_counters[i]);

            if (i != rank_me && to_send > 0) {
                // Use single rput for all kmers to this rank
                std::memcpy(send_ptrs[i].local(), kmers_by_rank[i].data() + sent_counters[i], to_send * sizeof(kmer_pair));
                sent_counters[i] += to_send;
            }
        }
        
        // Ensure all data transfers are complete
        upcxx::barrier();

        for (int j = 0; j < rank_n; ++j) {
            int i = (rank_me + j) % rank_n;
            
            size_t to_recv = std::min(seg_num_kmers_per_rank, recv_counts[i] - rcvd_counters[i]);

            if (i != rank_me && to_recv > 0) {
                kmer_pair kmers_rget[to_recv];

                upcxx::rget(recv_ptrs[i], kmers_rget, to_recv).wait();
                // Process each received kmer
                for (kmer_pair& kmer : kmers_rget) {
                    local_insert(kmer);
                }
                rcvd_counters[i] += to_recv;
            }
        }

        upcxx::barrier();
    }

    for (int i = 0; i < rank_n; ++i) {
        if (i != rank_me && send_counts[i] > 0) {
            upcxx::delete_array(send_ptrs[i]);
            send_ptrs[i] = nullptr;
        }
    }
}

// Local insert implementation using chaining
bool HashMap::local_insert(const kmer_pair& kmer) {
    uint64_t hash_val = kmer.hash();
    uint64_t slot = get_local_slot(hash_val);
    
    // Check if key already exists
    Node* current = buckets[slot];
    while (current != nullptr) {
        if (current->data.kmer == kmer.kmer) {
            // Key already exists, update the value
            current->data = kmer;
            return true;
        }
        current = current->next;
    }
    
    // Key doesn't exist, create a new node and add to the bucket
    Node* newNode = new Node(kmer);
    newNode->next = buckets[slot];
    buckets[slot] = newNode;
    
    return true;
}

// Find function
bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    if (key_kmer == pkmer_t()) {
        throw std::runtime_error("Error: key k-mer is empty!!!");
    }

    int target_rank = get_target_rank(key_kmer);
    
    if (target_rank == upcxx::rank_me()) {
        // Local find
        return local_find(key_kmer, val_kmer);
    } else {
        // Remote find using RPC
        auto result = upcxx::rpc(target_rank,
            [](const pkmer_t& key_kmer, upcxx::dist_object<HashMap*>& dobj) {
                kmer_pair val_kmer;
                bool found = (*dobj)->local_find(key_kmer, val_kmer);
                return std::make_pair(found, val_kmer);
            }, key_kmer, dobj).wait();
        
        if (result.first) {
            val_kmer = result.second;
            return true;
        }
        return false;
    }
}

// Local find implementation using chaining
bool HashMap::local_find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash_val = key_kmer.hash();
    uint64_t slot = get_local_slot(hash_val);
    
    // Traverse the linked list at the bucket
    Node* current = buckets[slot];
    while (current != nullptr) {
        if (current->data.kmer == key_kmer) {
            val_kmer = current->data;
            return true;
        }
        current = current->next;
    }
    
    return false;
}

// void HashMap::batch_find_kmers(const std::vector<pkmer_t>& key_kmers, std::vector<kmer_pair> & results) {
//     for (size_t i = 0; i < key_kmers.size(); ++i) {
//         kmer_pair val_kmer;
//         if (find(key_kmers[i], val_kmer)) {
//             results[i] = val_kmer;
//         } else {
//             throw std::runtime_error("Error: k-mer not found in hashmap at index " + std::to_string(i));
//         }
//     }
// }

void HashMap::batch_find_kmers(const std::vector<pkmer_t>& key_kmers, std::vector<kmer_pair> & results) {
    int rank_n = upcxx::rank_n();
    int rank_me = upcxx::rank_me();
    
    // Organize kmers by target rank
    std::vector<std::vector<pkmer_t>> kmers_by_rank(rank_n);
    std::vector<std::vector<size_t>> original_indices_by_rank(rank_n); // To track original positions
    
    for (size_t i = 0; i < key_kmers.size(); ++i) {
        const auto& kmer = key_kmers[i];
        int target_rank = get_target_rank(kmer);
        kmers_by_rank[target_rank].push_back(kmer);
        original_indices_by_rank[target_rank].push_back(i);
    }

    std::vector<upcxx::future<std::vector<kmer_pair>>> rpcs;
    std::vector<int> remote_ranks;
    for (int i = 0; i < rank_n; ++i) {
        if (i != rank_me && !kmers_by_rank[i].empty()) {
            remote_ranks.push_back(i);
            rpcs.push_back(upcxx::rpc(i,
                [](const std::vector<pkmer_t> key_kmers, upcxx::dist_object<HashMap*>& dobj) -> std::vector<kmer_pair> {
                    std::vector<kmer_pair> val_kmers;
                    for (int i = 0; i < key_kmers.size(); ++i) {
                        kmer_pair val_kmer;
                        bool found = (*dobj)->local_find(key_kmers[i], val_kmer);
                        if (found) {
                            val_kmers.push_back(val_kmer);
                        } else {
                            val_kmers.push_back(kmer_pair()); // Push empty kmer_pair if not found
                        }
                    }
                    return val_kmers;
                }, kmers_by_rank[i], dobj));
        }
    }

    // Process local kmers
    for (size_t i = 0; i < kmers_by_rank[rank_me].size(); ++i) {
        const auto& kmer = kmers_by_rank[rank_me][i];
        kmer_pair val_kmer;
        if (local_find(kmer, val_kmer)) {
            results[original_indices_by_rank[rank_me][i]] = val_kmer;
        }
        else {
            throw std::runtime_error("Error: Rank " + std::to_string(rank_me) + " k-mer not found in hashmap at local index " + std::to_string(i));
        }
    }

    // Process results from remote ranks
    for (int i = 0; i < remote_ranks.size(); ++i) {
        int rank = remote_ranks[i];
        std::vector<kmer_pair> remote_results = rpcs[i].wait();
        for (size_t j = 0; j < remote_results.size(); ++j) {
            // Check if the kmer is empty before assigning
            if (remote_results[j].kmer == pkmer_t()) {
                throw std::runtime_error("Error: Rank " + std::to_string(rank_me) + " k-mer not found in hashmap at rank "  + std::to_string(rank) +
                    " remote index " + std::to_string(j));
            }
            results[original_indices_by_rank[rank][j]] = remote_results[j];
        }
    }
}