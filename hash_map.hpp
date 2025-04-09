#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <vector>

struct HashMap {
    // Vector of global pointers, each pointing to an array on a different rank
    std::vector<upcxx::global_ptr<kmer_pair>> data_segments;
    std::vector<upcxx::global_ptr<int>> used_segments;

    // Local size of hash table segment for each rank
    size_t local_size;
    
    // Total distributed hash table size
    size_t total_size;

    // Constructor for distributed hash table
    HashMap(size_t total_table_size) {
        // Calculate local size for each rank
        local_size = total_table_size / upcxx::rank_n();
        total_size = total_table_size;

        // Allocate segments for each rank
        for (int i = 0; i < upcxx::rank_n(); ++i) {
            // Allocate data segment
            auto data_seg = upcxx::make_global<kmer_pair>(local_size);
            data_segments.push_back(data_seg);

            // Allocate used flag segment
            auto used_seg = upcxx::make_global<int>(local_size);
            used_segments.push_back(used_seg);
        }
    }

    // Determine which rank owns a particular slot
    int get_owner_rank(uint64_t slot) {
        return slot / local_size;
    }

    // Get local slot within a rank's segment
    uint64_t get_local_slot(uint64_t slot) {
        return slot % local_size;
    }

    // Distributed insert method
    bool insert(const kmer_pair& kmer) {
        uint64_t hash = kmer.hash();
        uint64_t probe = 0;
        bool success = false;

        while (!success && probe < total_size) {
            // Calculate global slot
            uint64_t global_slot = (hash + probe) % total_size;
            
            // Determine rank and local slot
            int owner_rank = get_owner_rank(global_slot);
            uint64_t local_slot = get_local_slot(global_slot);

            // Try to atomically claim the slot
            auto slot_ptr = used_segments[owner_rank] + local_slot;
            
            // Use a remote atomic operation to try and claim the slot
            int expected = 0;
            bool claimed = upcxx::atomic_compare_exchange(slot_ptr, expected, 1)
                .wait();

            if (claimed) {
                // Slot successfully claimed, write the data
                auto data_ptr = data_segments[owner_rank] + local_slot;
                upcxx::rput(kmer, data_ptr).wait();
                success = true;
            }

            ++probe;
        }

        return success;
    }

    // Distributed find method
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
        uint64_t hash = key_kmer.hash();
        uint64_t probe = 0;
        bool found = false;

        while (!found && probe < total_size) {
            // Calculate global slot
            uint64_t global_slot = (hash + probe) % total_size;
            
            // Determine rank and local slot
            int owner_rank = get_owner_rank(global_slot);
            uint64_t local_slot = get_local_slot(global_slot);

            // Check if slot is used
            auto used_ptr = used_segments[owner_rank] + local_slot;
            int is_used = upcxx::rget(used_ptr).wait();

            if (is_used) {
                // Retrieve the k-mer
                auto data_ptr = data_segments[owner_rank] + local_slot;
                val_kmer = upcxx::rget(data_ptr).wait();

                // Check if it matches the key
                if (val_kmer.kmer == key_kmer) {
                    found = true;
                }
            }

            ++probe;
        }

        return found;
    }

    // Cleanup method to free distributed memory
    void cleanup() {
        for (auto& seg : data_segments) {
            upcxx::deallocate(seg);
        }
        for (auto& seg : used_segments) {
            upcxx::deallocate(seg);
        }
    }
};