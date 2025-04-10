#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <list>
#include <vector>
#include <iostream> // Add for debugging

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
    
    // New functions for the new insert approach
    void process_kmers(const std::vector<kmer_pair>& kmers);
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
        if (recv_ptrs[i] != nullptr) {
            upcxx::delete_array(recv_ptrs[i]);
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

// Insert function - now just calls process_kmers with a single kmer
bool HashMap::insert(const kmer_pair& kmer) {
    std::vector<kmer_pair> kmers = {kmer};
    process_kmers(kmers);
    return true;
}

// Process kmers using the new approach
void HashMap::process_kmers(const std::vector<kmer_pair>& kmers) {
    int rank_n = upcxx::rank_n();
    int rank_me = upcxx::rank_me();
    
    // Step 1: Organize kmers by target rank
    std::vector<std::vector<kmer_pair>> kmers_by_rank(rank_n);
    
    for (const auto& kmer : kmers) {
        int target_rank = get_target_rank(kmer.kmer);
        if (target_rank >= 0 && target_rank < rank_n) {
            kmers_by_rank[target_rank].push_back(kmer);
        }
    }
    
    // Step 2: Process local kmers first for immediate progress
    for (const auto& kmer : kmers_by_rank[rank_me]) {
        local_insert(kmer);
    }
    
    // Step 3: Calculate counts for sending and receiving
    for (int i = 0; i < rank_n; ++i) {
        send_counts[i] = kmers_by_rank[i].size();
    }
    
    // Reset receive counts
    std::fill(recv_counts.begin(), recv_counts.end(), 0);
    
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
    
    // Step 4: Allocate memory for receiving kmers
    for (int i = 0; i < rank_n; ++i) {
        if (i != rank_me && recv_counts[i] > 0) {
            // Clean up any previous allocation
            if (recv_ptrs[i] != nullptr) {
                upcxx::delete_array(recv_ptrs[i]);
                recv_ptrs[i] = nullptr;
            }
            
            // Allocate memory for receiving
            recv_ptrs[i] = upcxx::new_array<kmer_pair>(recv_counts[i]);
            
            // Send the pointer to the sender rank
            if (recv_ptrs[i] != nullptr) {
                upcxx::rpc(i, 
                    [](upcxx::global_ptr<kmer_pair> ptr, int target_rank, upcxx::dist_object<HashMap*>& dobj) {
                        (*dobj)->send_ptrs[target_rank] = ptr;
                    }, recv_ptrs[i], rank_me, dobj).wait();
            }
        }
    }
    
    // Ensure all ranks have received their pointers
    upcxx::barrier();
    
    // Step 5: Use non-blocking rput to send kmers
    std::vector<upcxx::future<>> rputs;
    for (int i = 0; i < rank_n; ++i) {
        if (i != rank_me && !kmers_by_rank[i].empty()) {
            if (send_ptrs[i] == nullptr) {
                // Fall back to RPC if no destination pointer
                for (const auto& kmer : kmers_by_rank[i]) {
                    upcxx::rpc(i,
                        [](const kmer_pair& kmer, upcxx::dist_object<HashMap*>& dobj) {
                            (*dobj)->local_insert(kmer);
                        }, kmer, dobj).wait();
                }
                
                continue;
            }
            
            // Use single rput for all kmers to this rank
            rputs.push_back(upcxx::rput(
                kmers_by_rank[i].data(),  // source pointer
                send_ptrs[i],             // destination pointer
                kmers_by_rank[i].size()   // count
            ));
        }
    }
    
    // Step 6: Wait for all rputs to complete
    if (!rputs.empty()) {
        upcxx::when_all(rputs.begin(), rputs.end()).wait();
    }
    
    // Ensure all data transfers are complete
    upcxx::barrier();
    
    // Step 7: Process received kmers
    for (int i = 0; i < rank_n; ++i) {
        if (i != rank_me && recv_counts[i] > 0 && recv_ptrs[i] != nullptr) {
            // Process each received kmer
            for (size_t j = 0; j < recv_counts[i]; ++j) {
                kmer_pair kmer = upcxx::rget(recv_ptrs[i] + j).wait();
                local_insert(kmer);
            }
            
            // Free memory
            upcxx::delete_array(recv_ptrs[i]);
            recv_ptrs[i] = nullptr;
        }
    }
    
    // Final synchronization
    upcxx::barrier();
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