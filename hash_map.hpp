#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <vector>

struct HashMap {
    // Local storage
    std::vector<kmer_pair> data;
    std::vector<int> used;
    
    // Size of the local portion
    size_t my_size;
    
    // Distributed object reference for this hash map
    upcxx::dist_object<upcxx::global_ptr<HashMap>> local_ptr;
    
    // Constructor
    HashMap(size_t size);
    
    // Size functions
    size_t size() const noexcept;
    size_t local_size() const noexcept;
    
    // Calculate local slot from global hash
    uint64_t get_local_slot(uint64_t hash_val, uint64_t probe) const noexcept;
    
    // Get the target rank for a key
    int get_target_rank(uint64_t hash_val) const noexcept;
    
    // Main API functions
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);
    
    // Local operations
    bool local_insert(const kmer_pair& kmer);
    bool local_find(const pkmer_t& key_kmer, kmer_pair& val_kmer);
    
    // Helper functions
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);
    bool request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
};

HashMap::HashMap(size_t size) 
    : local_ptr(upcxx::make_global(this)) {
    // Calculate the size for each rank
    int rank_n = upcxx::rank_n();
    int rank_me = upcxx::rank_me();
    
    // Evenly distribute the hash table (with remainder to first ranks)
    size_t base_size = size / rank_n;
    size_t remainder = size % rank_n;
    
    my_size = base_size + (rank_me < remainder ? 1 : 0);
    
    // Allocate local storage
    data.resize(my_size);
    used.resize(my_size, 0);
    
    // Ensure all processes have initialized
    upcxx::barrier();
}

size_t HashMap::size() const noexcept {
    // Calculate total size across all ranks
    int rank_n = upcxx::rank_n();
    size_t total_size = 0;
    
    // Gather sizes from all ranks
    std::vector<size_t> all_sizes(rank_n);
    all_sizes[upcxx::rank_me()] = my_size;
    
    for (int i = 0; i < rank_n; i++) {
        size_t rank_size = upcxx::broadcast(all_sizes[i], i).wait();
        total_size += rank_size;
    }
    
    return total_size;
}

size_t HashMap::local_size() const noexcept {
    return my_size;
}

uint64_t HashMap::get_local_slot(uint64_t hash_val, uint64_t probe) const noexcept {
    // Determine the local slot index
    return (hash_val + probe) % my_size;
}

int HashMap::get_target_rank(uint64_t hash_val) const noexcept {
    // Determine which rank owns this hash value
    return hash_val % upcxx::rank_n();
}

bool HashMap::insert(const kmer_pair& kmer) {
    uint64_t hash_val = kmer.hash();
    int target_rank = get_target_rank(hash_val);
    
    if (target_rank == upcxx::rank_me()) {
        // Local insert
        return local_insert(kmer);
    } else {
        // Remote insert using RPC
        return upcxx::rpc(target_rank, 
            [](const kmer_pair& kmer, upcxx::dist_object<upcxx::global_ptr<HashMap>>& local_ptr) {
                HashMap* local_map = local_ptr->local();
                return local_map->local_insert(kmer);
            }, kmer, local_ptr).wait();
    }
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash_val = key_kmer.hash();
    int target_rank = get_target_rank(hash_val);
    
    if (target_rank == upcxx::rank_me()) {
        // Local find
        return local_find(key_kmer, val_kmer);
    } else {
        // Remote find using RPC
        auto result = upcxx::rpc(target_rank,
            [](const pkmer_t& key_kmer, upcxx::dist_object<upcxx::global_ptr<HashMap>>& local_ptr) {
                HashMap* local_map = local_ptr->local();
                kmer_pair val_kmer;
                bool found = local_map->local_find(key_kmer, val_kmer);
                return std::make_pair(found, val_kmer);
            }, key_kmer, local_ptr).wait();
        
        if (result.first) {
            val_kmer = result.second;
            return true;
        }
        return false;
    }
}

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
            // If we find an empty slot, the key is not present
            break;
        }
    } while (probe < my_size);
    
    return success;
}

// Helper functions
bool HashMap::slot_used(uint64_t slot) { 
    return used[slot] != 0; 
}

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) { 
    data[slot] = kmer; 
}

kmer_pair HashMap::read_slot(uint64_t slot) { 
    return data[slot]; 
}

bool HashMap::request_slot(uint64_t slot) {
    // Use atomic compare-exchange, with proper error handling
    // Initialize with 0 (empty)
    int expected = 0;
    
    // Try to set to 1 (used) if it's currently 0
    auto result = upcxx::atomic_compare_exchange_strong(
        &used[slot], 
        expected, 
        1
    ).wait();
    
    // If the operation returned 0, it succeeded (was empty, now marked as used)
    return (result == 0);
}