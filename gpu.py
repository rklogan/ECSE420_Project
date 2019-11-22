from numba import cuda
import numpy as np
import project
import math

WRITE_TO_FILE = False
OUTPUT_TO_CONSOLE = False
num_threads = 512

#finds the average of each row in a matrix
@cuda.jit
def parallel_averages(ip, op, num_items, offset):
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + offset
    if index >= num_items: return

    num_elts = len(ip[index])
    actual_elts = 0
    tmp = 0.0
    i = 0
    for i in range(num_elts):
        if ip[index][i] >= 0:
            tmp += ip[index][i]
            actual_elts += 1
    op[index] = tmp / actual_elts

#searches for the highest value in a subset of a vector
@cuda.jit
def parallel_search(ip, op, search_length, num_searchers, num_items, offset):
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + offset
    if index >= num_searchers: return

    input_idx = search_length * index
    best = -1.0
    best_index = -1
    for i in range(input_idx, input_idx + search_length):
        if index + i >= num_items: return

        if ip[index + i] > best:
            best = ip[index + i]
            best_index = index + i
    op[index] = best_index

"""                           MAIN                                        """
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


#set up and take the averages in parallel
ratings_array_d = cuda.to_device(ratings_array)
averages_d = cuda.device_array((num_items,), dtype=np.float)

for offset in range(0, num_items, num_threads):
    parallel_averages[num_blocks,threads_per_block](ratings_array_d, averages_d, num_items, offset)

averages = averages_d.copy_to_host()

#output the data
if WRITE_TO_FILE:
    with open('averages.txt', 'w+') as f:
        for i in range(num_items):
            to_write = str(asins[i]) + str(averages[i]) + '\n'
            f.write(to_write + '\n')
if OUTPUT_TO_CONSOLE:
    for i in range(num_items):
        to_write = str(asins[i]) + str(averages[i])
        print(to_write)

#setup and parallel search
search_length = 100
num_searchers = math.ceil(num_items / search_length)
best_averages_d = cuda.device_array((num_searchers,), dtype=np.float)

for offset in range(0, num_searchers, num_threads):
    parallel_search[num_blocks,threads_per_block](averages_d, best_averages_d, search_length, num_searchers, num_items, offset)

best_averages = best_averages_d.copy_to_host()

#do a final search of the remaining space
best_idx = -1
best = -1.0
for i in best_averages:
    x = int(i)
    if averages[x] > best:
        best = averages[x]
        best_idx = x

#output results
tmp = str(asins[best_idx])
best_item = project.item_dict[tmp]

to_write = best_item.asin + ' ' + best_item.brand + ' ' +  best_item.name + ' ' + str(averages[best_idx])
print(to_write)




