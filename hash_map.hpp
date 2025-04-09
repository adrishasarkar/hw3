#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

struct HashMap {
    std::vector<kmer_pair> data;
    std::vector<int> used;

    size_t my_size;
    size_t total_size;

    upcxx::dist_object<HashMap*> dobj;

    HashMap(size_t size);

    size_t size() const noexcept;
    size_t local_size() const noexcept;


    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);
    // Local operations (for this rank's portion)
    bool local_insert(const kmer_pair& kmer);
    bool local_find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
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
    
    // Resize local storage
    data.resize(my_size);
    used.resize(my_size, 0);
    
    // Ensure all ranks are initialized before proceeding
    upcxx::barrier();
}

// Get total hash table size
size_t HashMap::size() const noexcept {
    return total_size;
}

/ Get local portion size
size_t HashMap::local_size() const noexcept {
    return my_size;
}

// Determine which rank owns this key
int HashMap::get_target_rank(const pkmer_t& key_kmer) const {
    return key_kmer.hash() % upcxx::rank_n();
}

// Calculate local slot from hash and probe
uint64_t HashMap::get_local_slot(uint64_t hash_val, uint64_t probe) const {
    return (hash_val + probe) % my_size;
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

// Local insert implementation
bool HashMap::local_insert(const kmer_pair& kmer) {
    uint64_t hash_val = kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    
    do {
        uint64_t slot = get_local_slot(hash_val, probe++);
        success = request_slot(slot);
        if (success) {
            write_slot(slot, kmer);
        }
    } while (!success && probe < my_size);
    
    return success;
}

// Helper functions
bool HashMap::request_slot(uint64_t local_idx) {
    if (used[local_idx] != 0) {
        return false;
    } else {
        used[local_idx] = 1;
        return true;
    }
}

bool HashMap::slot_used(uint64_t local_idx) {
    return used[local_idx] != 0;
}

void HashMap::write_slot(uint64_t local_idx, const kmer_pair& kmer) {
    data[local_idx] = kmer;
}

kmer_pair HashMap::read_slot(uint64_t local_idx) {
    return data[local_idx];
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

// Local find implementation
bool HashMap::local_find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash_val = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    
    do {
        uint64_t slot = get_local_slot(hash_val, probe++);
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
                break;
            }
        } else {
            // If we hit an empty slot, the key is not in the table
            break;
        }
    } while (probe < my_size);
    
    return success;
}