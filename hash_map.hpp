#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

struct HashMap {
    upcxx::global_ptr<kmer_pair> data;
    upcxx::global_ptr<int64_t> used;

    // Atomic domain for synchronization
    upcxx::atomic_domain<int64_t> ad;

    size_t my_size;

    // Constructor
    HashMap(size_t size) : 
        ad({upcxx::atomic_op::load, upcxx::atomic_op::compare_exchange}) {
        my_size = size;
        data = upcxx::new_array<kmer_pair>(my_size);
        used = upcxx::new_array<int64_t>(my_size);

        // Initialize used flags to 0
        for (size_t i = 0; i < my_size; ++i) {
            upcxx::rput(0, used + i).wait();
        }
    }

    size_t size() const noexcept { 
        return my_size; 
    }

    // Insert method using atomic compare-exchange
    bool insert(const kmer_pair& kmer) {
        uint64_t hash = kmer.hash();
        uint64_t probe = 0;
        bool success = false;

        while (!success && probe < size()) {
            uint64_t slot = (hash + probe) % size();
            
            // Atomic compare-exchange
            int64_t expected = 0;
            int64_t desired = 1;
            
            success = ad.compare_exchange(
                used + slot,     // target location
                expected,         // expected value
                desired,          // desired value
                upcxx::memory_order::memory_order_relaxed
            ).wait();

            if (success) {
                // Write the k-mer to the claimed slot
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
            
            // Check if slot is used
            int64_t is_used = ad.load(
                used + slot, 
                upcxx::memory_order::memory_order_relaxed
            ).wait();

            if (is_used) {
                // Retrieve the k-mer
                val_kmer = upcxx::rget(data + slot).wait();

                // Check if it matches the key
                if (val_kmer.kmer == key_kmer) {
                    found = true;
                }
            }

            ++probe;
        }

        return found;
    }

    // Destructor
    ~HashMap() {
        // Destroy atomic domain
        ad.destroy();

        // Free allocated memory
        upcxx::delete_array(data);
        upcxx::delete_array(used);
    }
};