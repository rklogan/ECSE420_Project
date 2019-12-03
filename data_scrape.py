# Python 3 make sure to run in a terminal: pip3 install bs4 pandas
# Python 2 make sure to run in a terminal: pip install bs4 pandas
from bs4 import BeautifulSoup
from lxml import html
import pandas as pd
import requests
import re
import sys
import time

items = []
reviews = []

session = requests.session()
session.proxies = {}
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding": "gzip, deflate",
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT": "1", "Connection": "close", "Upgrade-Insecure-Requests": "1"}

urlBase = "https://www.amazon.ca"
bestsellers = requests.get(urlBase+"/bestsellers", headers=headers)
#bestsellers = requests.get("https://www.amazon.ca/product-reviews/B07PDHT5XP/",headers=headers)

soup = BeautifulSoup(bestsellers.content, 'html.parser')
# print(str(soup))
# sys.exit()
bestsellerstree = html.fromstring(str(soup))
productUrls = bestsellerstree.xpath('//a[@class="a-link-normal"]/@href')

print(productUrls)
print("\n\n")

# sys.exit()

count = 0
for url in productUrls:
    if count % 2 == 0:
        try:
            # time.sleep(2)
            item_req = requests.get(
                urlBase+productUrls[count], headers=headers)
            soup = BeautifulSoup(item_req.content, 'html.parser')
            item_tree = html.fromstring(str(soup))
            # print(item_0.text)
            asin = (url.split("/")[3])[:url.split("/")[3].find("?")]
            brand = item_tree.xpath(
                '//a[@id="bylineInfo"]/text()')[0].strip().rstrip()
            title = item_tree.xpath(
                '//span[@id="productTitle"]/text()')[0].strip().rstrip()
            url = urlBase+url
            image = item_tree.xpath(
                '//img[contains(@class,"a-dynamic-image")]/@src')[0]
            rating = float(item_tree.xpath(
                '//i[contains(@class,"a-icon-star")]/span[@class="a-icon-alt"]/text()')[0][:3])
            reviewUrl = urlBase+productUrls[count+1]
            totalReviews = int(item_tree.xpath(
                '//span[contains(@id,"acrCustomerReviewText")]/text()')[0].split(" ")[0].replace(",", ""))
            prices = "$"+item_tree.xpath('//span[contains(@class,"a-color-price")]/text()')[
                0].split("$")[1].strip().rstrip().replace(" ", "")
            print((asin, brand, title, url, image, rating,
                   reviewUrl, totalReviews, prices))
            items.append((asin, brand, title, url, image, rating,
                          reviewUrl, totalReviews, prices))
        except:
            items.append(("", "", "", "", "", 0.0, "", 0, "$0"))
        count += 1
    else:
        review_req = requests.get(urlBase+productUrls[count], headers=headers)
        soup = BeautifulSoup(review_req.content, 'html.parser')
        review_tree = html.fromstring(str(soup))
        # print(str(soup))

        # Get first few reviews (around 7 or 8 from testing)
        # Extend to more reviews using something that loads javascript
        # Typically have 10 reviews a page
        # https://www.amazon.com/product-reviews/B0000SX2UC/?pageNumber=2
        length = len(review_tree.xpath(
            '//span[contains(@class,"review-text-content")]/span[@class=""]/text()'))
        # print(length)
        for i in range(0, length):
            try:
                asin = (productUrls[count-1].split("/")
                        [3])[:url.split("/")[3].find("?")]
                name = review_tree.xpath(
                    '//span[@class="a-profile-name"]/text()')[i]
                rating = float(review_tree.xpath(
                    '//i[contains(@class,"a-icon-star")]/span[@class="a-icon-alt"]/text()')[i][:3])
                date = review_tree.xpath(
                    '//span[contains(@class,"review-date")]/text()')[i]
                ver = review_tree.xpath(
                    '//span[contains(@data-hook,"avp-badge")]/text()')[i]
                if ver.find("Verified") != -1:
                    verified = "true"
                else:
                    verified = "false"
                title = review_tree.xpath(
                    '//a[contains(@class,"review-title")]/span/text()')[i]
                body = review_tree.xpath(
                    '//span[contains(@class,"review-text-content")]/span[@class=""]/text()')[i]
                helpfulVotes = int(review_tree.xpath(
                    '//span[contains(@data-hook,"helpful-vote-statement")]/text()')[i].split(" ")[0].replace(",", ""))
                print((asin, name, rating, date, verified,
                       title, body, helpfulVotes))
                reviews.append((asin, name, rating, date,
                                verified, title, body, helpfulVotes))
            except:
                reviews.append(("", "", 0.0, "", "false", "", "", 0))
        count += 1


# for item in productUrls:
#    print(item.split("/"))

#x = re.split("\/",productUrls)
#x = re.findall("([^dp])(\w+)", productUrls[0])
# print(x)


df_items = pd.DataFrame(items, columns=[
                        "asin", "brand", "title", "url", "image", "rating", "reviewUrl", "totalReviews", "prices"])
df_items.to_csv('items_scraped.csv', index=False, encoding='utf-8')

df_reviews = pd.DataFrame(reviews, columns=[
                          "asin", "name", "rating", "date", "verified", "title", "body", "helpfulVotes"])
df_reviews.to_csv('reviews_scraped.csv', index=False, encoding='utf-8')
