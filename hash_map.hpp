// THIS WORKS. It is a serial implemetation with memory distrubution. working on making the parallel upgrades.

#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

struct HashMap {
    // UPC++ global pointers for distributed storage
    upcxx::global_ptr<kmer_pair> data;
    upcxx::global_ptr<int> used;

    // Size of the hash table
    size_t my_size;

    // Constructor
    HashMap(size_t size) {
        // Allocate global memory for data and used flags
        my_size = size;
        data = upcxx::new_array<kmer_pair>(my_size);
        used = upcxx::new_array<int>(my_size);

        // Initialize used flags to 0 on the local rank
        if (upcxx::rank_me() == 0) {
            for (size_t i = 0; i < my_size; ++i) {
                upcxx::rput(0, used + i).wait();
            }
        }
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
            
            // Check if slot is used
            int current_state = upcxx::rget(used + slot).wait();
            
            // If slot is unused, attempt to claim it
            if (current_state == 0) {
                // Try to put a "locked" state
                int locked_state = 1;
                upcxx::rput(locked_state, used + slot).wait();
                
                // Verify the slot is still unused
                int verify_state = upcxx::rget(used + slot).wait();
                
                if (verify_state == 1) {
                    // Successfully claimed, now write the k-mer
                    upcxx::rput(kmer, data + slot).wait();
                    success = true;
                    break;
                }
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
            int is_used = upcxx::rget(used + slot).wait();

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

    // Destructor to clean up global memory
    ~HashMap() {
        upcxx::delete_array(data);
        upcxx::delete_array(used);
    }
};