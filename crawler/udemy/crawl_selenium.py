from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
# chrome_options = Options()
# chrome_options.add_argument("--incognito")
# chrome_options.add_argument("--window-size=1920x1080")

def scroll_down(driver, count):
    """A method for scrolling the page."""

    # Get scroll height.
    last_height = driver.execute_script("return document.body.scrollHeight")

    i = 0
    while i < count:
        i += 1
        # Scroll down to the bottom.
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load the page.
        time.sleep(3)

        # Calculate new scroll height and compare with last scroll height.
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:

            break

        last_height = new_height

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome( executable_path="/home/hhn21/Documents/crawler/chromedriver_linux64/chromedriver")
url = "https://www.amazon.co.jp/gcx/%E3%83%99%E3%83%93%E3%83%BC-0-2%E6%AD%B3/gfhz/?categoryId=baby-neutral&scrollState=eyJpdGVtSW5kZXgiOjAsInNjcm9sbE9mZnNldCI6NTA1LjIwMzEyNX0%3D&sectionManagerState=eyJzZWN0aW9uVHlwZUVuZEluZGV4Ijp7ImFtYWJvdCI6MH19"

driver.get(url)
time.sleep(3)
els = driver.find_elements_by_css_selector("figure")

product = els


last_height = driver.execute_script("return document.body.scrollHeight")
i = 0
while i < 20:
    i += 1
    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        new_height = driver.execute_script("return document.body.scrollHeight")
    
        if new_height == last_height:
            break
        els = driver.find_elements_by_css_selector("figure")
        product.extend(els)
        last_height = new_height
    except:
        continue

# this is for testing
# product = product[0:2]

product_urls = []
for p in product:
    try:
        link = p.find_elements_by_tag_name('a')[0]
        url = link.get_attribute("href")
        product_urls.append(url)
    except:
        continue

review_pages = []
for p in product_urls:
    driver.get(p)
    try:
        url = driver.find_elements_by_xpath('//*[@id="reviews-medley-footer"]/div[2]/a')[0].get_attribute("href")
        review_pages.append(url)
    except:
        continue
        
with open("crawl_amazon_baby_gifts_0513.txt", "w+") as out:
    for page in review_pages:
        driver.get(page)
        while True:
            next_page = driver.find_elements_by_xpath('//*[@id="cm_cr-pagination_bar"]/ul/li[2]/a')
            if len(next_page) > 0:
                list_reviews = driver.find_elements_by_xpath("//*[contains(@id,'customer_review-')]")
                try:
                    for x in list_reviews:
                        time.sleep(2)
                        t = x.get_attribute('id')
                        reviews = driver.find_elements_by_xpath('//*[@id="' + t + '"]/div[4]/span/span')[0]
                        out.write("".join(reviews.text))
                        print(reviews.text)
                    next_page[0].click()
                    time.sleep(2)
                except:
                    continue
            else:
                break
#/div[4]/span/span

# scroll(driver, 5, 10)
# # time.sleep(2)
# # //*[@id="gcx-gf-section-0"]
# els = driver.find_elements_by_xpath("//*[contains(@id,'gcx-gf-section-')]")
# product = []
# for x in els:
#     time.sleep(2)
#     t = x.get_attribute('id')
#     print(t)
#     for i in range(1, 20):
#         time.sleep(1)
#         try:
#             path = driver.find_elements_by_xpath('//*[@id="' + t + '"]/div/section/div[' + str(i) + ']/figure/a')[0]
#             link = path.get_attribute('href')
#             print(link)
#         except:
#             print(driver.find_elements_by_xpath('//*[@id="' + t + '"]/div/section/div[' + str(i) + ']/figure/a'))
#             continue

# print(els)
#
# sections = [el]

# storyUrls = [el.get_attribute("href") for el in elements]
# elements = driver.find_elements_by_css_selector(".score")
# scores = [el.text for el in elements]
# elements = driver.find_elements_by_css_selector(".sitebit a")
# sites = [el.get_attribute("href") for el in elements]