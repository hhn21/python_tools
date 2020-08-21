from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
# chrome_options = Options()
# chrome_options.add_argument("--incognito")
# chrome_options.add_argument("--window-size=1920x1080")
from webdriver_manager.chrome import ChromeDriverManager
driver = webdriver.Chrome(ChromeDriverManager().install())

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

url = r"https://namegen.jp/?sex=male&country=japan&middlename=&middlename_cond=fukumu&middlename_rarity=&middlename_rarity_cond=ika&lastname=&lastname_cond=fukumu&lastname_rarity=&lastname_rarity_cond=ika&lastname_type=&firstname=&firstname_cond=fukumu&firstname_rarity=&firstname_rarity_cond=ika&firstname_type="

begin_time = time.time()

p = 0
while p < 5:
    #visit every pages
    p += 1
    print(p)
    try:
        driver.get(url)
    except:
        break
    #get the courses div
    names = driver.find_elements_by_class_name('name')
    print(names)
    out_names = []
    for name in names:
        try:
            first_name = name.find_elements_by_tag_name('a')[0].text
            last_name = name.find_elements_by_tag_name('a')[1].text
            out_names.append(first_name + last_name)
        except:
            continue
    with open("names.txt", "a+") as out:
        for name in out_names:
            out.write(name)
            out.write("\n")

    print('***')
    print('names got  : ', len(out_names))
    print('Time taken : ', time.time() - begin_time)
    print('***')

with open("names.txt", "r") as file:
    names = [name.rstrip('\n') for name in file]

filtered_names = set(names)
with open("filtered_names.txt", "a+") as out:
        for name in filtered_names:
            out.write(name)
            out.write("\n")