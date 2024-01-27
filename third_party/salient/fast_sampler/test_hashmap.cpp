#include "parallel_hashmap/phmap.h"
#include <iostream>
#include <vector>
#include <time.h>
#include <numeric>

/*
 * given a lot of node features, we cache 10% of all features
 * at each batch, we need to filter and identify the cached entries
 *
 */
const int NUM_NODES = 100000000; // 100M 
const int NUM_CACHES = 10000000; // 10M
const int BATCH_SIZE = 1000000; // 1M
//const int NUM_NODES = 10000000; // 10M 
//const int NUM_CACHES = 1000000; // 1M
//const int BATCH_SIZE = 100000; // 0.1M

const int num_batches = 1000;

int main() {
    phmap::flat_hash_map<int64_t, int64_t> cache_id_map;
    cache_id_map.reserve(NUM_CACHES);
    srand((unsigned int) time(0));
    // generate cache map id
    clock_t begin = clock();
    int cnt = 0;
    while (cache_id_map.size() < NUM_CACHES) {
        int tmp = rand() % NUM_NODES;
        if (cache_id_map.find(tmp) == cache_id_map.end())
            cache_id_map[tmp] = cnt++;
    }
    clock_t end = clock();
    std::cout << "generating cache map time: " << double(end-begin) / CLOCKS_PER_SEC  << "s" << std::endl;

    begin = clock();
    std::vector<bool> cache_id_flag;
    cache_id_flag.reserve(NUM_NODES);
    for (auto & n : cache_id_map)
        cache_id_flag[n.first] = true;
    end = clock();
    std::cout << "generating cache flag time: " << double(end-begin) / CLOCKS_PER_SEC  << "s" << std::endl;

    // generate fake batch and time
    std::vector<int> time_elapsed(num_batches);
    for (int i = 0; i <  num_batches; ++i) {
        std::vector<int> cur_batch;
        cur_batch.reserve(BATCH_SIZE);
        for (int j = 0; j < BATCH_SIZE; ++j) cur_batch.push_back(rand()%NUM_NODES);
        begin = clock();
        std::vector<int> cur_filtered;
        cur_filtered.reserve(BATCH_SIZE);
        for (auto n : cur_batch) {
            //if (cache_id_map.find(n) != cache_id_map.end()) {
            if (cache_id_flag[n]) {
                //cur_filtered.push_back(n);
                cnt += 1;
            }
        }
        end = clock();
        time_elapsed.push_back(end-begin);
    }
    std::cout << std::accumulate(time_elapsed.begin(), time_elapsed.end(), 0.0) / (float)num_batches / (CLOCKS_PER_SEC/1000) << " ms" << std::endl;

    return 0;
}
