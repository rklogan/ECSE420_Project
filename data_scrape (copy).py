from requests_html import HTMLSession
from lxml import html
import pandas as pd
import requests
import re
import sys
import time

session = HTMLSession()

items=[('a','b',1,2,'a1','b',1,2,2)]
reviews=[('a','b',1,2,'a1','b',1,2)]

session = requests.session()
session.proxies = {}
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}

urlBase = "https://www.amazon.ca"
bestsellers = requests.get(urlBase+"/bestsellers",headers=headers)
#print(r.text)
bestsellerstree = html.fromstring(bestsellers.content)
productUrls = bestsellerstree.xpath('//a[@class="a-link-normal"]/@href')

print(productUrls)
print("\n\n")

#sys.exit()

count = 0
for url in productUrls:
    if count%2 == 0:
        #time.sleep(2)
        item_req = requests.get(urlBase+productUrls[count],headers=headers)
        item_tree = html.fromstring(item_req.content)
        #print(item_0.text)
        asin=(url.split("/")[3])[:url.split("/")[3].find("?")]
        brand=item_tree.xpath('//a[@id="bylineInfo"]/text()')[0]
        title=item_tree.xpath('//span[@id="productTitle"]/text()')[0].strip().rstrip()
        url=urlBase+url
        image=item_tree.xpath('//img[contains(@class,"a-dynamic-image")]/@src')[0]
        rating=float(item_tree.xpath('//i[contains(@class,"a-icon-star")]/span[@class="a-icon-alt"]/text()')[0][:3])
        reviewUrl=urlBase+productUrls[count+1]
        totalReviews=int(item_tree.xpath('//span[contains(@id,"acrCustomerReviewText")]/text()')[0].split(" ")[0].replace(",",""))
        prices="$"+item_tree.xpath('//span[contains(@class,"a-color-price")]/text()')[0].split("$")[1].strip().rstrip().replace(" ","")
        print((asin,brand,title,url,image,rating,reviewUrl,totalReviews,prices))        
        items.append((asin,brand,title,url,image,rating,reviewUrl,totalReviews,prices))
        count+=1
    else:
        review_req = requests.get(urlBase+productUrls[0],headers=headers)
        review_tree = html.fromstring(review_req.content)
	print(review_req.text)
        asin=(productUrls[count-1].split("/")[3])[:url.split("/")[3].find("?")]
        name=review_tree.xpath('//span[@class="a-profile-name"]/text()')[0]
        rating=float(review_tree.xpath('//i[contains(@class,"a-icon-star")]/span[@class="a-icon-alt"]/text()')[0][:3])
        date=review_tree.xpath('//span[contains(@class,"review-date")]/text()')[0]
	ver=review_tree.xpath('//span[contains(@data-hook,"avp-badge")]/text()')[0]
	if ver.find("Verified") != -1:
            verified="true"
        else:
            verified="false"
        title=review_tree.xpath('//a[contains(@class,"review-title")]/span/text()')[0]
        body=review_tree.xpath('//div[contains(@class,"review-text-content")]/span[@class=""]/text()')[0]
        helpfulVotes=int(review_tree.xpath('//span[contains(@data-hook,"helpful-vote-statement")]/text()')[0].split(" ")[0])
        print((asin,name,rating,date,verified,title,body,helpfulVotes))        
        sys.exit()
        reviews.append((asin,name,rating,date,verified,title,body,helpfulVotes))
        count+=1


for item in productUrls:
    print(item.split("/"))

x = re.split("\/",productUrls)
x = re.findall("([^dp])(\w+)", productUrls[0])
print(x)

# 10 reviews a page
# https://www.amazon.com/product-reviews/B0000SX2UC/?pageNumber=2


df_items = pd.DataFrame(items, columns=["asin","brand","title","url","image","rating","reviewUrl","totalReviews","prices"])
df_items.to_csv('items_scraped.csv', index=False)

df_reviews = pd.DataFrame(reviews, columns=["asin","name","rating","date","verified","title","body","helpfulVotes"])
df_reviews.to_csv('reviews_scraped.csv', index=False)
