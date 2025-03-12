import os
import time
import pandas as pd
import duckdb
import undetected_chromedriver as uc
from selenium.webdriver import ChromeOptions
from bs4 import BeautifulSoup
from tqdm import tqdm
from fastparquet import write
import requests

# Constants
WRITE_BATCH_SIZE = 10
SELENIUM_TIMEOUT = 15

# Headers and Cookies for requests
HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-DE,en;q=0.9,de-DE;q=0.8,de;q=0.7,en-US;q=0.6",
    "priority": "u=0, i",
    "sec-ch-ua": '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133")',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
}

COOKIES = {
    "prov": "6fb614ed-8d51-4787-be72-c9863376e0a8",
    "OptanonAlertBoxClosed": "2024-12-01T09:43:15.703Z",
    "eupubconsent-v2": "CQI8_lgQI8_lgAcABBENBSF4AP_AAEPAACiQKiQBgAGAAgAHQAaABecFQwVEAvOAGACAAdAC8wAA.f_gACHgAAAAA",
    "_so_tgt": "1ac0e1f6-a9a7-4634-8d80-6248ab7b92d6",
    "usr": "p=[160;BountyEndingSoon;Bounty;]",
    "_cfuvid": "mZF2nEV48sTyizyCV_wCs8tLLP1O.sgqIwMg0gvazmU-1739035389029-0.0.1.1-604800000",
    "notice-jbn-2": "4;1739035394810",
    "_ga_S812YQPLT2": "GS1.1.1739043810.4.1.1739043961.0.0.0",
    "__cflb": "0H28vFHtoAR1ohjxFgDZBwZ5H5NWURXnsdCZTFELbdE",
    "_gid": "GA1.2.2027663712.1739131441",
    "acct": "t=ODpWBQP/2aAtrcgjWdqsj4FLQost41CE&s=gOgc/23Dl4kl3O0fkibiTaouFnxbE+qX",
    "__cf_bm": "glTr_hzj.ZqauNU5QtUZ1ue4HOvybziIW61R2PmaHrg-1739198898-1.0.1.1-ZiNG98IT9tzfvV67r5gmpx2nClYDElTXYFVrekuDCJPqAPEn79lTBGex.GaThovG5.hvdtoyGWeneUa9iWg3FQ",
    "cf_clearance": "HznrVHMMXE7aHY56FjrXxQsP29n0XVf6h6918YwTqLU-1739198901-1.2.1.1-yE1CFDQbVpRmc.z24qPZSRp4VwpyRpBD8UGQkjOypkfrM5psICTpt24H2K2Iq4LFNxSzNcdQXaGwH5NsWRgVTIlZJoD9osxbhCaz5Rxet1E9a5sle6WgEYwcTsILFB7iSWokuM0ZTZmrC2X41atSdHJmjyCxArkQiYk5nP9vtOj.2SAw60PpEsFIuaXx_BUO1FQ8PFevIlgI.5zaO9AtrEORNh567FQ2P_OkTWS8PIixF9r9tldQCa_RxyHW8PAv.toZe3nDiaGSYt1TYJOdTlYAf9MX8EUMJfPJ.iZB8WE",
    "_gat": "1",
    "_ga_WCZ03SZFCQ": "GS1.1.1739197032.38.1.1739198902.56.0.0",
    "_ga": "GA1.1.1013657177.1733128174"
}

def configure_driver():
    """
    Configure and return a Selenium WebDriver.

    Returns:
        WebDriver: Configured Selenium WebDriver instance.
    """
    chrome_options = ChromeOptions()
    # Uncomment the following options for headless mode and faster performance
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")

    driver = uc.Chrome(options=chrome_options)
    driver.set_page_load_timeout(SELENIUM_TIMEOUT)
    driver.set_script_timeout(SELENIUM_TIMEOUT)
    return driver

def append_to_parquet(dataframe, path):
    """
    Append to an existing Parquet file or create a new one if it doesn't exist.

    Args:
        dataframe (pd.DataFrame): The DataFrame to append.
        path (str): The path to the Parquet file.
    """
    try:
        if os.path.exists(path):
            write(path, dataframe, append=True)
        else:
            write(path, dataframe)
    except Exception as e:
        print(f"Error writing to Parquet file '{path}': {e}")

# Scraping Functions
def scrape_with_selenium(driver, question_id):
    url = f'https://stackoverflow.com/posts/{question_id}/timeline'
    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        return parse_timeline_html(soup)
    except Exception as e:
        print(f"Error scraping with Selenium for question ID {question_id}: {e}")
        return [], []


def scrape_with_requests(question_id):
    time.sleep(0.2)
    url = f'https://stackoverflow.com/posts/{question_id}/timeline?filter=NoVoteDetail'
    try:
        response = requests.get(url, headers=HEADERS, cookies=COOKIES)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return parse_timeline_html(soup)
        else:
            print(f"Non-200 response for question ID {question_id}: {response.status_code}")
    except Exception as e:
        print(f"Error scraping with Requests for question ID {question_id}: {e}")
    return [], []


def parse_timeline_html(soup):
    accepted_timestamps, users = [], []
    bounty_rows = soup.find_all('tr', attrs={'data-eventtype': 'vote', 'class': lambda x: x and 'datehash' in x})
    for row in bounty_rows:
        event_cell = row.find('td', class_='wmn1')
        date_cell = row.find('span', class_='relativetime')
        user_cell = row.find_all('td', class_='ws-nowrap')[1] if len(
            row.find_all('td', class_='ws-nowrap')) > 1 else None

        if event_cell and date_cell:
            event_text = event_cell.get_text(strip=True).lower()
            date = date_cell.get('title')

            if 'accept' in event_text:
                accepted_timestamps.append(date)
                user_link = user_cell.find('a') if user_cell else None
                user_name = user_link.get_text(strip=True) if user_link else 'Anonymous'
                users.append(user_name)
    return accepted_timestamps, users


def main(data_dir, query_results_path, results_path, method='selenium'):
    query_results_df = pd.read_csv(query_results_path)
    question_ids = query_results_df['QuestionId'].tolist()
    results = []

    if method == 'selenium':
        driver = configure_driver()
        scrape_func = lambda qid: scrape_with_selenium(driver, qid)
    else:
        scrape_func = scrape_with_requests

    with tqdm(total=len(question_ids), desc="Scraping Progress") as pbar:
        for question_id in question_ids:
            accepted_timestamps, users = scrape_func(question_id)
            if accepted_timestamps:
                for timestamp, user in zip(accepted_timestamps, users):
                    results.append({'question_id': question_id, 'accepted_timestamp': timestamp, 'user': user})

            if len(results) >= WRITE_BATCH_SIZE:
                df = pd.DataFrame(results)
                append_to_parquet(df, results_path)
                results = []

            pbar.update(1)

    if results:
        df = pd.DataFrame(results)
        append_to_parquet(df, results_path)

    if method == 'selenium':
        driver.quit()

if __name__ == "__main__":
    DATA_DIR = "./Data"
    QUERY_RESULTS_PATH = os.path.join(DATA_DIR, "QueryResults.csv")
    RESULTS_PATH = os.path.join(DATA_DIR, "acceptedanswers.parquet")

    main(DATA_DIR, QUERY_RESULTS_PATH, RESULTS_PATH, method='selenium')