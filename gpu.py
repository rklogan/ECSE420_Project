from numba import cuda
import numpy as np
import project
import math

WRITE_TO_FILE = True
OUTPUT_TO_CONSOLE = True
num_threads = 512

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


ratings_array_d = cuda.to_device(ratings_array)
averages_d = cuda.device_array((num_items,), dtype=np.float)

for offset in range(0, num_items, num_threads):
    parallel_averages[num_blocks,num_threads](ratings_array_d, averages_d, num_items, offset)

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

#copy the data from device
averages = averages_d.copy_to_host()



