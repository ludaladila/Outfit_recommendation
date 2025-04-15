import pandas
from selenium.webdriver.common.by import By
from selenium import webdriver
import re

d = []
chrome_option = webdriver.ChromeOptions()
chrome_option.add_argument("--disable-blink-features=AutomationControlled")
chrome_option.add_experimental_option("excludeSwitches", ['enable-automation', 'enable-logging'])
browser = webdriver.Chrome(options=chrome_option)
browser.maximize_window()

grnders = {'women': {'Tops & T-Shirts': 'https://www2.hm.com/en_us/women/products/tops.html', 'Jeans': 'https://www2.hm.com/en_us/women/products/jeans.html', 'Pants': 'https://www2.hm.com/en_us/women/products/pants.html', 'Shirts & Blouses': 'https://www2.hm.com/en_us/women/products/shirts-blouses.html', 'Shorts': 'https://www2.hm.com/en_us/women/products/shorts.html', 'Blazers & Vests': 'https://www2.hm.com/en_us/women/products/blazers-vests.html', 'Cardigans & Sweaters': 'https://www2.hm.com/en_us/women/products/cardigans-sweaters.html', 'Jackets & Coats': 'https://www2.hm.com/en_us/women/products/jackets-coats.html', 'Hoodies & Sweatshirts': 'https://www2.hm.com/en_us/women/products/hoodies-sweatshirts.html', 'Skirts': 'https://www2.hm.com/en_us/women/products/skirts.html'},
     'men': {'T-shirts & Tops': 'https://www2.hm.com/en_us/men/products/t-shirts-tank-tops.html', 'Shirts': 'https://www2.hm.com/en_us/men/products/shirts.html', 'Polos': 'https://www2.hm.com/en_us/men/products/polos.html', 'Jeans': 'https://www2.hm.com/en_us/men/products/jeans.html', 'Pants': 'https://www2.hm.com/en_us/men/products/pants.html', 'Suits & Blazers': 'https://www2.hm.com/en_us/men/products/suits-blazers.html', 'Hoodies & Sweatshirts': 'https://www2.hm.com/en_us/men/products/hoodies-sweatshirts.html', 'Sweaters & Cardigans': 'https://www2.hm.com/en_us/men/products/cardigans-sweaters.html', 'Jackets & Coats': 'https://www2.hm.com/en_us/men/products/jackets-coats.html', 'Shorts': 'https://www2.hm.com/en_us/men/products/shorts.html'}
     }
women = {'Tops & T-Shirts': 'https://www2.hm.com/en_us/women/products/tops.html', 'Jeans': 'https://www2.hm.com/en_us/women/products/jeans.html', 'Pants': 'https://www2.hm.com/en_us/women/products/pants.html', 'Shirts & Blouses': 'https://www2.hm.com/en_us/women/products/shirts-blouses.html', 'Shorts': 'https://www2.hm.com/en_us/women/products/shorts.html', 'Blazers & Vests': 'https://www2.hm.com/en_us/women/products/blazers-vests.html', 'Cardigans & Sweaters': 'https://www2.hm.com/en_us/women/products/cardigans-sweaters.html', 'Jackets & Coats': 'https://www2.hm.com/en_us/women/products/jackets-coats.html', 'Hoodies & Sweatshirts': 'https://www2.hm.com/en_us/women/products/hoodies-sweatshirts.html', 'Skirts': 'https://www2.hm.com/en_us/women/products/skirts.html'}
mon = {'T-shirts & Tops': 'https://www2.hm.com/en_us/men/products/t-shirts-tank-tops.html', 'Shirts': 'https://www2.hm.com/en_us/men/products/shirts.html', 'Polos': 'https://www2.hm.com/en_us/men/products/polos.html', 'Jeans': 'https://www2.hm.com/en_us/men/products/jeans.html', 'Pants': 'https://www2.hm.com/en_us/men/products/pants.html', 'Suits & Blazers': 'https://www2.hm.com/en_us/men/products/suits-blazers.html', 'Hoodies & Sweatshirts': 'https://www2.hm.com/en_us/men/products/hoodies-sweatshirts.html', 'Sweaters & Cardigans': 'https://www2.hm.com/en_us/men/products/cardigans-sweaters.html', 'Jackets & Coats': 'https://www2.hm.com/en_us/men/products/jackets-coats.html', 'Shorts': 'https://www2.hm.com/en_us/men/products/shorts.html'}
d = []
for grnder in grnders:
    for category in grnders[grnder]:
        c = 0
        for page in range(1, 7):
            print(grnders[grnder][category] + f'?page={page}', end=' ')
            browser.get(grnders[grnder][category] + f'?page={page}')
            for i in range(100):
                if len(browser.find_elements(by=By.XPATH, value='//*[@id="products-listing-section"]/ul/li')) != 0:
                    break
            print(len(browser.find_elements(by=By.XPATH, value='//*[@id="products-listing-section"]/ul/li')))
            for i in browser.find_elements(by=By.XPATH, value='//*[@id="products-listing-section"]/ul/li'):
                item = {}
                item['grnder'] = grnder
                item['category'] = category
                item['href'] = i.find_element(by=By.TAG_NAME, value='a').get_attribute('href')
                d.append(item)
                c += 1
                if c == 200:
                    break
            if len(browser.find_elements(by=By.CLASS_NAME, value='b395d2.f6b03d.a08eca.fe373a.dfc6c7.fc59b7.ec32a0')) != 0 and page != 1:
                break
d1 = []
pandas.DataFrame(d).to_excel('data1.xlsx', index=False)
for item in d:
    browser.get(item['href'])
    while True:
        if len(browser.find_elements(by=By.XPATH, value='//*[@id="product-reco-swg"]/div/ul/li')) != 0:
            break
        if len(browser.find_elements(by=By.ID, value='hm-error-title')) != 0:
            browser.refresh()
    item['product_name'] = browser.find_element(by=By.CLASS_NAME, value='fa226d.af6753.d582fb').text
    item['price'] = browser.find_element(by=By.CLASS_NAME, value='edbe20.ac3d9e.d9ca8b').text
    item['description'] = re.findall('<p class="d1cd7b ca7db2 e2b79d">(.*?)</p>', browser.page_source)[0]
    item['main_image_url'] = browser.find_element(by=By.XPATH, value='//*[@id="__next"]/main/div[2]/div/div/div[2]/div/div/div[3]/section/div[1]/div/a[1]/div[2]/span/img').get_attribute('src')
    n = 1
    browser.execute_script("arguments[0].scrollIntoView();", browser.find_element(by=By.XPATH, value='//*[@id="product-reco-swg"]/div/ul/li'))
    while True:
        if len(browser.find_elements(by=By.XPATH, value='//*[@id="product-reco-swg"]/div/ul/li')[0].find_element(by=By.CLASS_NAME, value='c63693').get_attribute('src')) != 0:
            break
    for i in browser.find_elements(by=By.XPATH, value='//*[@id="product-reco-swg"]/div/ul/li')[:2]:
        item[f'suggestion_{n}_name'] = i.find_element(by=By.CLASS_NAME, value='fa226d.ca21a4.f25ed4.b5233a').text
        item[f'suggestion_{n}_image'] = i.find_element(by=By.CLASS_NAME, value='c63693').get_attribute('src')
        item[f'suggestion_{n}_price'] = i.find_element(by=By.CLASS_NAME, value='aeecde.ac3d9e').text
        n += 1
    print(item)
    d1.append(item)
pandas.DataFrame(d1).to_excel('data.xlsx', index=False)
