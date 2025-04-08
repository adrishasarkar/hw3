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
    
    // Array of global pointers to all hash maps, one per rank
    upcxx::global_ptr<HashMap> *rank_map;
    
    // Constructor and destructor
    HashMap(size_t size);
    ~HashMap();
    
    // Size functions
    size_t size() const noexcept;
    size_t local_size() const noexcept;
    
    // Get the target rank for a key
    int get_target_rank(const pkmer_t& key_kmer) const noexcept;
    
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

HashMap::HashMap(size_t size) {
    // Calculate the size for each rank
    int rank_n = upcxx::rank_n();
    size_t base_size = size / rank_n;
    size_t remainder = size % rank_n;
    
    // My rank gets base_size + 1 if remainder > my_rank
    my_size = base_size;
    if (upcxx::rank_me() < remainder) {
        my_size++;
    }
    
    // Allocate local storage
    data.resize(my_size);
    used.resize(my_size, 0);
    
    // Create a distributed object directory
    rank_map = new upcxx::global_ptr<HashMap>[rank_n];
    
    // Get a global pointer to this local map
    upcxx::global_ptr<HashMap> my_ptr = upcxx::new_<HashMap>(this);
    
    // Share global pointer with all ranks using collective communication
    for (int i = 0; i < rank_n; i++) {
        rank_map[i] = upcxx::broadcast(my_ptr, i).wait();
    }
    
    // Ensure all ranks have shared their pointers
    upcxx::barrier();
}

HashMap::~HashMap() {
    delete[] rank_map;
}

size_t HashMap::size() const noexcept {
    // Calculate the total size across all ranks
    int rank_n = upcxx::rank_n();
    size_t total = 0;
    
    for (int i = 0; i < rank_n; i++) {
        size_t rank_size = upcxx::rpc(i, 
            [](upcxx::global_ptr<HashMap> map_ptr) {
                return map_ptr.local()->local_size();
            }, rank_map[i]).wait();
        total += rank_size;
    }
    
    return total;
}

size_t HashMap::local_size() const noexcept {
    return my_size;
}

int HashMap::get_target_rank(const pkmer_t& key_kmer) const noexcept {
    // Use a better distribution hash function if needed
    return key_kmer.hash() % upcxx::rank_n();
}

bool HashMap::insert(const kmer_pair& kmer) {
    int target_rank = get_target_rank(kmer.kmer);
    
    if (target_rank == upcxx::rank_me()) {
        // Local insert
        return local_insert(kmer);
    } else {
        // Remote insert using RPC
        return upcxx::rpc(target_rank, 
            [](upcxx::global_ptr<HashMap> map_ptr, kmer_pair kmer) {
                return map_ptr.local()->local_insert(kmer);
            }, rank_map[target_rank], kmer).wait();
    }
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    int target_rank = get_target_rank(key_kmer);
    
    if (target_rank == upcxx::rank_me()) {
        // Local find
        return local_find(key_kmer, val_kmer);
    } else {
        // Remote find using RPC
        auto result = upcxx::rpc(target_rank,
            [](upcxx::global_ptr<HashMap> map_ptr, pkmer_t key_kmer) {
                HashMap* local_map = map_ptr.local();
                kmer_pair val_kmer;
                bool found = local_map->local_find(key_kmer, val_kmer);
                return std::make_pair(found, val_kmer);
            }, rank_map[target_rank], key_kmer).wait();
        
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
    
    // Linear probing within local portion
    do {
        uint64_t slot = (hash_val + probe++) % my_size;
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
    
    // Linear probing within local portion
    do {
        uint64_t slot = (hash_val + probe++) % my_size;
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
                break;
            }
        } else if (!slot_used(slot)) {
            // If we hit an empty slot, the key is not in the table
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
    // Atomic compare-and-swap to avoid race conditions
    if (used[slot] != 0) {
        return false;
    } else {
        // In UPC++, we should use atomic operations here
        // For simplicity using direct assignment, but consider atomic ops
        used[slot] = 1;
        return true;
    }
}