from numba import cuda
import numpy as np
import project
import math
import time

num_threads = 256

@cuda.jit
def find_max_parallel(dataset, maximums, row_of_maxes, rows_to_process, num_items):
    output_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if output_idx >= num_items: return

    max_found = -1.0
    max_row = -1

    num_elts = len(dataset[output_idx])

    starting_row = output_idx * rows_to_process
    current_row = starting_row
    end_row = starting_row + rows_to_process
    if end_row >= num_items:
        end_row = num_items

    while current_row < end_row:
        actual_data_points = 0
        acc = 0.0
        i = 0
        while i < num_elts:
            if dataset[current_row][i] <= 0.0:
                break
            else:
                acc += dataset[current_row][i]
                actual_data_points += 1
                i += 1
        ave = acc / actual_data_points

        if ave > max_found:
            max_found = ave
            max_row = current_row

        current_row += 1

    maximums[output_idx] = max_found
    row_of_maxes[output_idx] = max_row

if __name__ == "__main__":
    project.load_data()

    num_items = len(project.ratings_dict)

    #search for the item with the most review
    most_reviews = -1
    for ratings in project.ratings_dict.values():
        if len(ratings) > most_reviews:
            most_reviews = len(ratings)

    #generate itermediary lists so ordering can be preserved
    asins = list(project.ratings_dict.keys())
    ratings = list(project.ratings_dict.values())

    ratings_list = []
    for i in range(num_items):
        row = []
        for j in range(most_reviews):
            if j < len(ratings[i]):
                row.append(ratings[i][j])
            else:
                row.append(-1.0)
        ratings_list.append(row)
    ratings_array = np.array(ratings_list, dtype = np.float)

    #decide how much work each thread should do
    rows_per_thread = -1
    if num_threads >= len(ratings_array):
        num_threads = len(ratings_array)
        rows_per_thread = 1
    else:
        rows_per_thread = math.ceil(len(ratings_array) / num_threads)


    #decide how to allocate threads/blocks
    if num_threads > num_items:
        num_threads = num_items
    num_blocks = 1
    threads_per_block = num_threads
    max_threads_per_block = 32

    while threads_per_block > max_threads_per_block:
        num_blocks += 1
        threads_per_block = math.ceil(float(num_threads) / float(num_blocks))

        #check if we're using too many blocks
        if(num_blocks > 65535):
            num_blocks = 1
            threads_per_block = num_threads
            max_threads_per_block *= 2

    #device allocation
    ratings_array_d = cuda.to_device(ratings_array)
    rows_per_thread_d = cuda.to_device(rows_per_thread)
    num_items_d = cuda.to_device(num_items)
    maxes_d = cuda.device_array((num_threads,), dtype=np.float)
    max_indexes_d = cuda.device_array((num_threads,), dtype=np.int)

    #do parallel computation
    start = time.time()
    find_max_parallel[num_blocks,threads_per_block](ratings_array_d, maxes_d, max_indexes_d, rows_per_thread, num_items)
    end = time.time()

    #return the data
    maxes = maxes_d.copy_to_host()
    max_indexes = max_indexes_d.copy_to_host()

    print(maxes)

    #do a linear search on the values
    max_val = -1
    max_idx = -1
    for i in range(num_threads):
        if maxes[i] > max_val:
            max_val = maxes[i]
            max_idx = max_indexes[i]

    print('Best Asin: ' + str(asins[max_idx]))
    print('Rating: ' + str(max_val))
    print('Time Elapsed: ' + str(end-start))


