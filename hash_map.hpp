#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

struct HashMap {
    // UPC++ global pointers for distributed storage
    upcxx::global_ptr<kmer_pair[]> data;
    upcxx::global_ptr<int32_t[]> used;
    
    // Size of the hash table
    size_t my_size;
    
    // Atomic domain for atomic operations
    upcxx::atomic_domain<int32_t> ad;
    
    // Constructor
    HashMap(size_t size) {
        my_size = size;
        
        // Allocate global memory for data and used flags
        data = upcxx::new_array<kmer_pair>(my_size);
        used = upcxx::new_array<int32_t>(my_size);
        
        // Initialize used flags to 0 on the local rank
        if (upcxx::rank_me() == 0) {
            for (size_t i = 0; i < my_size; ++i) {
                upcxx::rput(0, used + i).wait();
            }
        }
        
        // Initialize atomic domain with needed operations
        ad = upcxx::atomic_domain<int32_t>({
            upcxx::atomic_op::load,
            upcxx::atomic_op::compare_exchange
        });
        
        upcxx::barrier();
    }
    
    // Size of the hash table
    size_t size() const noexcept {
        return my_size;
    }
    
    // Insert method
    bool insert(const kmer_pair& kmer) {
        uint64_t hash = kmer.hash();
        uint64_t probe = 0;
        bool success = false;
        
        while (!success && probe < size()) {
            uint64_t slot = (hash + probe) % size();
            
            // Atomically check if slot is unused (0) and try to set it to used (1)
            int32_t expected = 0;
            int32_t desired = 1;
            
            // Use compare_exchange to atomically claim the slot
            success = ad.compare_exchange(
                used + slot,
                expected,
                desired,
                std::memory_order_acq_rel
            ).wait();
            
            if (success) {
                // Successfully claimed, now write the k-mer
                upcxx::rput(kmer, data + slot).wait();
                break;
            }
            
            ++probe;
        }
        
        return success;
    }
    
    // Find method
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
        uint64_t hash = key_kmer.hash();
        uint64_t probe = 0;
        bool found = false;
        
        while (!found && probe < size()) {
            uint64_t slot = (hash + probe) % size();
            
            // Atomically check if slot is used
            int32_t is_used = ad.load(used + slot, std::memory_order_acquire).wait();
            
            if (is_used) {
                // Retrieve the k-mer
                val_kmer = upcxx::rget(data + slot).wait();
                
                // Check if it matches the key
                if (val_kmer.kmer == key_kmer) {
                    found = true;
                    break;
                }
            }
            
            ++probe;
        }
        
        return found;
    }
    
    // Destructor to clean up global memory
    ~HashMap() {
        // Destroy atomic domain before deallocating memory
        ad.destroy();
        
        upcxx::delete_array(data);
        upcxx::delete_array(used);
    }
};
