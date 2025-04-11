#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <upcxx/upcxx.hpp>
#include <vector>

#include "hash_map.hpp"
#include "kmer_t.hpp"
#include "read_kmers.hpp"

#include "butil.hpp"

int main(int argc, char** argv) {
    upcxx::init();

    // TODO: Dear Students,
    // // Please remove this if statement, when you start writing your parallel implementation.
    // if (upcxx::rank_n() > 1) {
    //     throw std::runtime_error("Error: parallel implementation not started yet!"
    //                              " (remove this when you start working.)");
    // }

    if (argc < 2) {
        BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test [prefix]]\n");
        upcxx::finalize();
        exit(1);
    }

    std::string kmer_fname = std::string(argv[1]);
    std::string run_type = "";

    if (argc >= 3) {
        run_type = std::string(argv[2]);
    }

    std::string test_prefix = "test";
    if (run_type == "test" && argc >= 4) {
        test_prefix = std::string(argv[3]);
    }

    int ks = kmer_size(kmer_fname);

    if (ks != KMER_LEN) {
        throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
                                 "-mers, while this binary is compiled for " +
                                 std::to_string(KMER_LEN) +
                                 "-mers.  Modify packing.hpp and recompile.");
    }

    size_t n_kmers = line_count(kmer_fname);

    // Load factor of 0.5
    size_t hash_table_size = n_kmers * (1.0 / 0.5);
    HashMap hashmap(hash_table_size);

    if (run_type == "verbose") {
        BUtil::print("Initializing hash table of size %d for %d kmers.\n", hash_table_size,
                     n_kmers);
    }

    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

    if (run_type == "verbose") {
        BUtil::print("Finished reading kmers.\n");
    }

    upcxx::barrier();

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<kmer_pair> start_nodes;

    // Process kmers in batches using the new approach
    hashmap.process_kmers(kmers);
    
    // Collect start nodes
    for (auto& kmer : kmers) {
        if (kmer.backwardExt() == 'F') {
            start_nodes.push_back(kmer);
        }
    }
    
    auto end_insert = std::chrono::high_resolution_clock::now();
    upcxx::barrier();

    double insert_time = std::chrono::duration<double>(end_insert - start).count();
    if (run_type != "test") {
        BUtil::print("Finished inserting in %lf\n", insert_time);
    }
    upcxx::barrier();

    auto start_read = std::chrono::high_resolution_clock::now();

    std::vector<std::list<kmer_pair>> contigs;
    std::vector<size_t> active_indices;

    for (const auto& start_kmer : start_nodes) {
        std::list<kmer_pair> contig;
        contig.push_back(start_kmer);
        contigs.push_back(std::move(contig));
        if (start_kmer.forwardExt() != 'F') {
            active_indices.push_back(contigs.size() - 1);
        }
    }

    while (!active_indices.empty()) {
        std::vector<pkmer_t> next_kmers;
        
        for (size_t idx : active_indices) {
            next_kmers.push_back(contigs[idx].back().next_kmer());
        }

        std::vector<kmer_pair> found_kmers(next_kmers.size(), kmer_pair());

        hashmap.batch_find_kmers(next_kmers, found_kmers);

        std::vector<size_t> new_active_indices;
        for (size_t i = 0; i < active_indices.size(); ++i) {
            size_t idx = active_indices[i];
            const kmer_pair &found = found_kmers[i];

            // if (found.kmer == pkmer_t()) {
            //     // If a kmer was not found, this is an error condition.
            //     throw std::runtime_error("Error: k-mer not found in hashmap for contig index " + std::to_string(idx));
            // }
            
            // Extend the contig with the newly found kmer.
            contigs[idx].push_back(found);
            
            // If the contig is still active (the new kmer's forward extension is not 'F'),
            // keep its index for the next round.
            if (found.forwardExt() != 'F') {
                new_active_indices.push_back(idx);
            }
        }

        active_indices.swap(new_active_indices);
    }

    auto end_read = std::chrono::high_resolution_clock::now();
    upcxx::barrier();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> read = end_read - start_read;
    std::chrono::duration<double> insert = end_insert - start;
    std::chrono::duration<double> total = end - start;

    int numKmers = std::accumulate(
        contigs.begin(), contigs.end(), 0,
        [](int sum, const std::list<kmer_pair>& contig) { return sum + contig.size(); });

    if (run_type != "test") {
        BUtil::print("Assembled in %lf total\n", total.count());
    }

    if (run_type == "verbose") {
        printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
               " (%lf read, %lf insert, %lf total)\n",
               upcxx::rank_me(), contigs.size(), numKmers, start_nodes.size(), read.count(),
               insert.count(), total.count());
    }

    if (run_type == "test") {
        std::ofstream fout(test_prefix + "_" + std::to_string(upcxx::rank_me()) + ".dat");
        for (const auto& contig : contigs) {
            fout << extract_contig(contig) << std::endl;
        }
        fout.close();
    }

    upcxx::finalize();
    return 0;
}
