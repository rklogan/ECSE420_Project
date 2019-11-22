import pandas as pd

items_filename = 'items.csv'
reviews_filename = 'reviews.csv'

class item:
    def __init__(self, id, b, n, rating=0):
        self.asin = id
        self.brand = b
        self.name = n
        self.rating = 0

def load_data():
    num_items = 0
    asin_col = -1
    brand_col = -1
    name_col = -1

    item_dict = {}
    items = pd.read_csv(items_filename)

    i = 0
    for heading in items:
        if 'asin' in heading:
            asin_col = i
        elif 'brand' in heading:
            brand_col = i
            print(brand_col)
        elif 'title' in heading:
            name_col = i
        i += 1

    for _, row in items.iterrows():
        asin = row[asin_col]
        row_item = item(asin, row[brand_col], row[name_col])
        item_dict[asin] = row_item
