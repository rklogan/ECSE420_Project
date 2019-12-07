import pandas as pd
import time


items_filename = './items.csv'
reviews_filename = './reviews.csv'
item_dict = {}
ratings_dict = {}

class item:
    def __init__(self, id, b, n, rating=0):
        self.asin = id
        self.brand = b
        self.name = n
        self.rating = 0

#load the data from file and stores it in the dictionaries
def load_data():
    num_items = 0
    asin_col = -1
    brand_col = -1
    name_col = -1
    rating_col = -1


    items = pd.read_csv(items_filename)

    i = 0
    for heading in items:
        if 'asin' in heading:
            asin_col = i
        elif 'brand' in heading:
            brand_col = i
        elif 'title' in heading:
            name_col = i
        i += 1

    for _, row in items.iterrows():
        asin = row[asin_col]
        row_item = item(asin, row[brand_col], row[name_col])
        item_dict[asin] = row_item

    reviews = pd.read_csv(reviews_filename)

    i = 0
    for heading in reviews:
        if 'asin' in heading:
            asin_col = i
        elif 'rating' in heading:
            rating_col = i
        i += 1


    start = time.time() #this line can be moved to go above wherever the code to parallelize is
    for _, row in reviews.iterrows():
        asin = row[asin_col]
        rating = row[rating_col]

        if not asin in ratings_dict:
            ratings_dict[asin] = [rating]
        else:
            ratings_dict[asin].append(rating)


    best_phone = find_highest(ratings_dict)

    best_asin = best_phone[0]

    bf_name = " "



    if best_asin in item_dict:
        bf_name = item_dict[best_asin].name
        

    end = time.time()

    print("best_phone: ", bf_name)
    print("rating: ", best_phone[1])
    print("time elapsed: ", round(end - start, 5))


def avg(lst):
    return sum(lst) / len(lst)


def find_highest(ratings_dict):
    avgs = {}
    highest_avg = 0
    asin = ""
    name = ""
    
    for i in ratings_dict:
        if i not in avgs:
            avgs[i] = round(avg(ratings_dict[i]), 4)


    for j in avgs:
        #print ("current = ", avgs[j])
        #print("highest_avg: ", highest_avg)
        if avgs[j] >= highest_avg:
            highest_avg = avgs[j]
            asin = j



    return (asin, highest_avg)
        
            





load_data()


