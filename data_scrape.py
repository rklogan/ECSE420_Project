# Python 3 make sure to run in a terminal: pip3 install bs4 pandas
# Python 2 make sure to run in a terminal: pip install bs4 pandas
from bs4 import BeautifulSoup
from lxml import etree, html
import pandas as pd
import requests
import re
import sys
import time
import traceback

# Store an array of items from Amazon and reviews for them.
items = []
reviews = []

# Create a session object to query Amazon URLs with.
session = requests.session()
session.proxies = {}

# Spoof our user agent to get around scraping restrictions.
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding": "gzip, deflate",
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT": "1", "Connection": "close", "Upgrade-Insecure-Requests": "1"}

# The base URL of the Amazon website we want to scrape.
urlBase = "https://www.amazon.ca"

# Get the content from the bestsellers page.
bestsellers = requests.get(urlBase+"/bestsellers", headers=headers)

# Execute some javascript on the bestsellers page.
soup = BeautifulSoup(bestsellers.content, 'html.parser')

# Parse HTML from the bestsellers page into a tree object.
bestsellerstree = html.fromstring(str(soup))

# Get a list of URLs to items and their reviews on the bestsellers page.
productUrls = bestsellerstree.xpath('//a[@class="a-link-normal"]/@href')

# Clean up the URLs.
for i in range(0,len(productUrls)):
    if i % 2 == 0:
        parts = productUrls[i].split("/")
        productUrls[i] = "/"+parts[1]+"/"+parts[2]+"/"+parts[3]+"/"
    else:
        if "product-reviews" not in productUrls[i]:
            productUrls.insert(i,"")
        else:
            parts = productUrls[i].split("/")
            productUrls[i] = "/"+parts[1]+"/"+parts[2]+"/"

# Print the item and review URLs for debugging.
print(productUrls)
print("Total urls to scrape: "+str(len(productUrls)))
print("\n\n")

# Retrieve attributes from each item and its reviews.
count = 0
for url in productUrls:
    if count % 2 == 0:
        try:
            # Request item URL.
            item_req = requests.get(
                urlBase+productUrls[count], headers=headers)
            soup = BeautifulSoup(item_req.content, 'html.parser')
            item_tree = html.fromstring(str(soup))

            # Obtain the following attributes from item URL: asin, brand, title, url, image, rating, reviewUrl, totalReviews, prices.
            asin = (url.split("/")[3])[:url.split("/")[3].find("?")]
            try:
                brand = item_tree.xpath('//a[@id="bylineInfo"]/text()')[0].strip().rstrip()
            except:
                brand = ""
            title = item_tree.xpath(
                '//span[@id="productTitle"]/text()')[0].strip().rstrip()
            url = str(urlBase+url)
            image = item_tree.xpath(
                '//img[contains(@class,"a-dynamic-image")]/@src')[0]
            rating = float(item_tree.xpath(
                '//i[contains(@class,"a-icon-star")]/span[@class="a-icon-alt"]/text()')[0][:3])
            reviewUrl = str(urlBase+productUrls[count+1])
            try:
                totalReviews = int(item_tree.xpath('//span[contains(@id,"acrCustomerReviewText")]/text()')[0].split(" ")[0].replace(",", ""))
            except:
                totalReviews = 0
            prices = "$"+item_tree.xpath('//span[contains(@class,"a-color-price")]/text()')[
                0].split("$")[1].strip().rstrip().replace(" ", "")

            # Debugging.
            #print((asin, brand, title, url, image, rating,
            #       reviewUrl, totalReviews, prices))

            # Append item attributes to final list.
            items.append((asin, brand, title, url, image, rating,
                          reviewUrl, totalReviews, prices))
        except Exception:
            traceback.print_exc()
            items.append(("", "", "", "", "", 0.0, "", 0, "$0"))
        count += 1
    else:
        if len(productUrls[count]) == 0:
            # Occurs when no reviews for item are available.
            count += 1
            continue
        else:
            # Request review URL.
            review_req = requests.get(urlBase+productUrls[count], headers=headers)
            soup = BeautifulSoup(review_req.content, 'html.parser')
            review_tree = html.fromstring(str(soup))

            # Get first page reviews (around 10 from testing or less depending on the item)
            # A possible improvement to get more reviews could be to use selenium or use url parameters:
            # e.g. https://www.amazon.com/product-reviews/B0000SX2UC/?pageNumber=2
            # Typically each page has 10 reviews for products with more than 10 reviews.
            
            # Obtain and store review texts that have been scraped.
            revs = []
            review_spans = review_tree.xpath(
            '//span[contains(@class,"review-text-content")]/span[@class=""]')
            for r in review_spans:
                revs.append(r.text_content())

            # Count the actual number of reviews (exclude ratings without reviews).
            totalTextReviews = len(revs)

            # Update the total reviews for the item.
            (a,b,t,u,i,r,rU,tR,p) = items[int(count/2)]
            items[int(count/2)] = (a,b,t,u,i,r,rU,totalTextReviews,p)

            # Scrape attributes from each review.
            for i in range(0, totalTextReviews):
                try:
                    # Obtain the following attributes from review URL: asin, name, rating, date, verified, title, body, helpfulVotes.
                    asin = (productUrls[count-1].split("/")
                            [3])[:url.split("/")[3].find("?")]
                    name = review_tree.xpath(
                        '//span[@class="a-profile-name"]/text()')[i]
                    rating = float(review_tree.xpath(
                        '//i[contains(@class,"a-icon-star")]/span[@class="a-icon-alt"]/text()')[i][:3])
                    date = review_tree.xpath(
                        '//span[contains(@class,"review-date")]/text()')[i]
                    try:                    
                        ver = review_tree.xpath('//span[contains(@data-hook,"avp-badge")]/text()')[i]
                    except:
                        ver = ""
                    if ver.find("Verified") != -1:
                        verified = "true"
                    else:
                        verified = "false"
                    title = review_tree.xpath(
                        '//a[contains(@class,"review-title")]/span/text()')[i]
                    body = review_tree.xpath(
                        '//span[contains(@class,"review-text-content")]/span[@class=""]/text()')[i]
                    try:
                        review_tree.xpath('//span[contains(@data-hook,"helpful-vote-statement")]/text()')
                        helpfulVotes = int(review_tree.xpath('//span[contains(@data-hook,"helpful-vote-statement")]/text()')[i].split(" ")[0].replace(",", ""))
                    except:
                        helpfulVotes = 0

                    # Debugging.
                    #print((asin, name, rating, date, verified,
                    #       title, body, helpfulVotes))

                    # Append review attributes to final list.
                    reviews.append((asin, name, rating, date,
                                    verified, title, body, helpfulVotes))
                except Exception:
                    traceback.print_exc()
                    reviews.append(("", "", 0.0, "", "false", "", "", 0))
            count += 1

# Store the retrieved items into a pandas dataframe and then items_scraped.csv
df_items = pd.DataFrame(items, columns=["asin", "brand", "title", "url", "image", "rating", "reviewUrl", "totalReviews", "prices"])
df_items.to_csv('items_scraped.csv', index=False, encoding='utf-8')

# Store the retrieved item reviews into a pandas dataframe and then reviews_scraped.csv
df_reviews = pd.DataFrame(reviews, columns=["asin", "name", "rating", "date", "verified", "title", "body", "helpfulVotes"])
df_reviews.to_csv('reviews_scraped.csv', index=False, encoding='utf-8')

print("Data scraped successfully.")
