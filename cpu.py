import project
import threading
import time

averages = {}


def average(ratings_dict):
    local_averages = {}
    for asin in ratings_dict:
        sum = 0
        asin_specific_dict = ratings_dict[asin]
        for rating in asin_specific_dict:
            sum += rating
        local_averages[asin] = sum / len(asin_specific_dict)

    asin = ""
    highest_avg = 0
    for i in local_averages:
        if local_averages[i] > highest_avg:
            highest_avg = local_averages[i]
            asin = i

    averages[asin] = highest_avg


if __name__ == "__main__":
    threads = []
    num_threads = 4

    project.load_data()
    start = time.time()
    x = project.ratings_dict

    number_of_asin = len(project.ratings_dict.keys())
    asin_per_thread = number_of_asin // num_threads
    subset_dict = {}

    for asin_counter, asin in enumerate(project.ratings_dict):
        if asin_counter % asin_per_thread == 0:
            subset_dict = {}

        subset_dict[asin] = project.ratings_dict[asin]

        if asin_counter == number_of_asin or (asin_counter != 0 and (asin_counter + 1) % asin_per_thread == 0):
            thread = threading.Thread(target=average, kwargs={
                'ratings_dict': subset_dict})
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()

    asin = ""
    highest_avg = 0
    for i in averages:
        if averages[i] > highest_avg:
            highest_avg = averages[i]
            asin = i

    best_phone = (asin, highest_avg)
    time_elapsed = time.time() - start

    print("best_phone: ", best_phone[0])
    print("rating: ", best_phone[1])
    print("time elapsed: ", time_elapsed)
