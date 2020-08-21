from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
# chrome_options = Options()
# chrome_options.add_argument("--incognito")
# chrome_options.add_argument("--window-size=1920x1080")

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome( executable_path="/home/hhn21/Documents/crawler/chromedriver_linux64/chromedriver")

begin_time = time.time()
review_count = 0
link_count = 0

# with open("udemy_links/test.txt", "r") as f:
#     links = f.read().splitlines()

with open("udemy_5.txt", "r") as f:
    links = [link.rstrip('\n') for link in f]

# remove duplication
courses_urls = list(set(links))
with open("udemy_reviews/udemy_5_reviews.txt", "a+") as out:
    for page in courses_urls:
        link_count += 1
        print(page)
        driver.get(page + '#review')
        time.sleep(5)
        print('Crawling reviews...', end='')

        while True:
            print('.', end='')
            next_page = driver.find_elements_by_xpath("//button[@data-purpose='show-more-review-button']")
            try:
                next_page[0].click()
                time.sleep(1)
            except:
                break
        list_reviews = driver.find_elements_by_xpath('//div[@data-purpose="review-comment-content"]/p')
        review_count += len(list_reviews)
        print('\nwriting to file...', end='')
        for review in list_reviews:
            out.write("".join(review.text))
            out.write("\n")
        print('***')
        print('Link visited   : ', link_count)
        print('Review crawled : ', review_count)
        print('Time tanken    : ', time.time() - begin_time)
        print('***')
