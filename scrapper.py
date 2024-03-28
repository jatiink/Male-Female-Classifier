import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import requests
import traceback
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from termcolor import colored
import time

driver_path = r"chromedriver.exe"  # Put driver path here


def getImageUrlsFromProductLink(p_url, product_id, save_path, headers):
    try:
        chr_options = Options()
        chr_options.add_experimental_option("detach", True)
        driver = webdriver.Chrome(driver_path, options=chr_options)

        time.sleep(2)
        try:
            driver.get(p_url)
        except Exception as e:
            print("------------------error---------------- : ".format(e))
            traceback.print_exc()
            driver.quit()
            time.sleep(15)
            chr_options = Options()
            chr_options.add_experimental_option("detach", True)
            driver = webdriver.Chrome(driver_path, options=chr_options)
            time.sleep(2)
            driver.get(p_url)

        time.sleep(2)
        data = driver.find_elements_by_class_name("image-grid-image")  # Put class here by inspecting the webpage
        for index, i in enumerate(data):
            if index == 0:
                image_url = i.get_attribute("style")
                image_url = image_url.split('url("')[1].split('");')[0]
                print(image_url)
                image_dir = save_path
                os.makedirs(image_dir, exist_ok=True)
                image_name = str(product_id) + '.jpg'
                image_path = os.path.join(image_dir, image_name)
                if not os.path.exists(image_path):
                    download(image_url, image_path)
        driver.quit()
    except Exception as e:
        driver.quit()
        print(colored("Error getting image urls from product page {}: {}".format(p_url, e), "red"))
        traceback.print_exc()


def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = requests.get(url)
        # write to file
        file.write(response.content)


def extraction(category, store_url, page_num, end_page, num_products, product_ls, save_path):
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}
    while page_num <= end_page:
        try:
            url = store_url + "&p=" + str(page_num)
            print("url: ", url)
            products = []
            print(colored("Page Number: " + str(page_num), "blue"))
            print(colored("Page_url: " + url, "blue"))
            chr_options = Options()
            chr_options.add_experimental_option("detach", True)
            driver = webdriver.Chrome(driver_path, options=chr_options)
            time.sleep(2)
            try:
                driver.get(url)
            except Exception as e:
                print("------------------error---------------- in page get : ".format(e))
                traceback.print_exc()
                driver.quit()
                time.sleep(30)
                chr_options = Options()
                chr_options.add_experimental_option("detach", True)
                driver = webdriver.Chrome(driver_path, options=chr_options)
                time.sleep(2)
                driver.get(url)

            time.sleep(2)
            data = driver.find_elements_by_tag_name("a")    # element from inspecting webpage
            for i in data:
                if i.get_attribute("target") == "_blank":  # attibutes from inspecting webpage
                    products.append(i.get_attribute("href"))
            driver.quit()
            products = list(set(products))

            print(colored("Total products on page: " + str(len(products)), "blue"))
            if len(products) <= 0:
                break
            else:
                for product in products:
                    if product not in product_ls:
                        p_url = product
                        print(colored("Product url :" + p_url, "green"))
                        product_id = p_url.split("/")[-2]
                        getImageUrlsFromProductLink(p_url, product_id, save_path, headers)
                        num_products += 1
                        product_ls.append(product)
            page_num += 1
        except Exception as e:
            page_num += 1
            print(colored("Error on List Page {}: {} ".format(url, e), "red"))
            traceback.print_exc()
    return (num_products, product_ls)


store_url = r"https://www."  # put website here
save_path = r"D:\data\test"  # put path here where you want to save

page_num = 1
end_page = 1

product_ls = []
ls = [""]

for items in ls:
    num_products = 0
    category = items
    num_products, product_ls = extraction(category, store_url, page_num, end_page, num_products, product_ls, save_path)
logs_path = r"D:\data\logs"  # creating logs  
os.makedirs(logs_path, exist_ok=True)
with open(os.path.join(logs_path, "men_suits_logs.txt"), 'w+') as f:
    for item in product_ls:
        f.write(f"{item}\n")
