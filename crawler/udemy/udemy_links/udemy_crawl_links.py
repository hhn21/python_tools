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

url = r"https://www.udemy.com/courses/teaching-and-academics/?lang=ja&sort=popularity"

begin_time = time.time()
link_count = 0

p = 0
while p < 32:
    #visit every pages
    p += 1
    next_page = url + '&p={}'.format(p)
    print(next_page)
    try:
        driver.get(next_page)
        time.sleep(5)
    except:
        break
    #get the courses div
    courses = driver.find_elements_by_class_name('list-view-course-card--course-card-wrapper--TJ6ET')
    
    #extract url out of the div
    courses_url = []
    for c in courses:
        try:
            link = c.find_elements_by_tag_name('a')[0]
            courses_url.append(link.get_attribute("href"))
        except:
            continue
    with open("udemy_teaching_links.txt", "a+") as out:
        for page in courses_url:
            link_count += 1
            out.write(page)
            out.write("\n")

    print('***')
    print('Link visited: ', link_count)
    print('Time taken : ', time.time() - begin_time)
    print('***')
