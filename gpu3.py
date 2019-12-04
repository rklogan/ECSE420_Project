import pandas as pd
import math
import numpy as np
import numba
from numba import cuda
import time
import sys

reviews_filename = 'reviews.csv'
parallel_search_size = 12

reviews = []
indexes = []
asin_lookup = {}
id_lookup = {}
def load_ratings():
    reviews_panda = pd.read_csv(reviews_filename)

    i = 0
    for _, row in reviews_panda.iterrows():
        if not row['asin'] in id_lookup:
            id_lookup[row['asin']] = i
            asin_lookup[i] = row['asin']
            i += 1
        reviews.append(float(row['rating']))
        indexes.append(id_lookup[row['asin']])

@cuda.jit
def parallel_average(indexes, reviews, output, offset):
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + offset
    if index >= len(output): return

    output[index] = np.float(0.0)

    #get the sum of all reviews with my index
    i = 0
    count = 0
    while i < len(indexes):
        if indexes[i] == index:
            output[index] += reviews[index]

            count += 1
        i += 1

    #compute the average
    output[index] = output[index] / count

def run(nt = 1024):
    start = time.time()

    #decide how to allocate threads/blocks
    num_blocks = 1
    threads_per_block = nt
    max_threads_per_block = 32

    while threads_per_block > max_threads_per_block:
        num_blocks += 1
        threads_per_block = math.ceil(float(nt) / float(num_blocks))

        #check if we're using too many blocks
        if(num_blocks > 65535):
            num_blocks = 1
            threads_per_block = nt
            max_threads_per_block *= 2

    #convert to numpy array
    reviews_array = np.array(reviews, dtype=np.float)
    index_array = np.array(indexes, dtype=np.int)

    d_reviews_array = cuda.to_device(reviews_array)
    d_index_array = cuda.to_device(index_array)
    d_output = cuda.device_array((len(id_lookup,)), dtype=np.float)
    d_best_index = cuda.device_array((parallel_search_size,), dtype=np.int)

    #compute the averages in parallel
    for offset in range(0, len(id_lookup), nt):
        parallel_average[num_blocks, threads_per_block](d_index_array, d_reviews_array, d_output, offset)

    output = d_output.copy_to_host()

    print(time.time() - start)

    i = 0
    max_val = -1.0
    max_id = 0
    while i < len(output):
        if output[i] > max_val:
            max_val = output[i]
            max_id = i
        i += 1

    print(asin_lookup[max_id])

if __name__ == "__main__":
    nt = 1024
    if len(sys.argv) >= 2:
        nt = int(sys.argv[1])

    load_ratings()
    run(nt)
