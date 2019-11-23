import project
import threading
import time

averages = {}


def average(asin, ratings):
    sum = 0
    for rating in ratings:
        sum += rating

    averages[asin] = sum / len(ratings)


if __name__ == "__main__":
    i = 0
    threads = []
    num_threads = 128
    threadcounter = 0

    project.load_data()
    start = time.time()
    x = project.ratings_dict
    for asin in project.ratings_dict:
        if threadcounter < num_threads:
            threadcounter += 1
            thread = threading.Thread(target=average, kwargs={
                'asin': asin,
                'ratings': project.ratings_dict[asin]})
            threads.append(thread)
            thread.start()

        else:
            for thread in threads:
                thread.join()
                threadcounter = 0

    asin = ""
    highest_avg = 0
    for i in averages:
        if averages[i] >= highest_avg:
            highest_avg = averages[i]
            asin = i

    best_phone = (asin, highest_avg)
    time_elapsed = time.time() - start

    print("best_phone: ", best_phone[0])
    print("rating: ", best_phone[1])
    print("time elapsed: ", time_elapsed)
