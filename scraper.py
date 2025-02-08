import os
import time
import pandas as pd
import duckdb
import undetected_chromedriver as uc
from selenium.webdriver import ChromeOptions
from bs4 import BeautifulSoup
from tqdm import tqdm
from fastparquet import write

# Constants
DATA_DIR = "./data"
RESULTS_PATH = os.path.join(DATA_DIR, "bounty_timeline_results.parquet")
PROCESSED_IDS_PATH = os.path.join(DATA_DIR, "processed_question_ids.parquet")
BOUNTY_VOTES_PATH = os.path.join(DATA_DIR, "BountyVotes.parquet")
WRITE_BATCH_SIZE = 10
SELENIUM_TIMEOUT = 15

# Utility Functions
def create_data_directory():
    """
    Ensure the data directory exists.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_processed_ids():
    """
    Load already processed question IDs from the Parquet file.

    Returns:
        list: A list of processed question IDs.
    """
    if os.path.exists(PROCESSED_IDS_PATH):
        try:
            processed_ids_df = pd.read_parquet(PROCESSED_IDS_PATH, engine="fastparquet")
            return processed_ids_df['question_id'].tolist()
        except Exception as e:
            print(f"Error reading processed IDs file: {e}")
    return []

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

def get_unprocessed_post_ids():
    con = duckdb.connect()
    query = f"""
        SELECT DISTINCT b.PostId
        FROM parquet_scan('{BOUNTY_VOTES_PATH}') b
        WHERE b.VoteTypeId = 8
          AND b.PostId NOT IN (
              SELECT DISTINCT r.question_id
              FROM parquet_scan('{RESULTS_PATH}') r
          )
    """
    post_ids_data = con.execute(query).fetchall()
    con.close()
    return [row[0] for row in post_ids_data]

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

def scrape_timeline_events(driver, question_id):
    """
    Scrape bounty start/end events for a given question ID.

    Args:
        driver (WebDriver): Selenium WebDriver instance.
        question_id (int): The question ID to scrape.

    Returns:
        dict: A dictionary containing question ID, bounty start, and bounty end dates.
    """
    time.sleep(1)
    url = f'https://stackoverflow.com/posts/{question_id}/timeline'

    try:
        driver.get(url)
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')

        bounty_start, bounty_end = [], []
        bounty_rows = soup.find_all(
            'tr',
            attrs={
                'data-eventtype': 'history',
                'class': lambda x: x and 'datehash' in x
            }
        )

        for row in bounty_rows:
            event_cell = row.find('td', class_='wmn1')
            date_cell = row.find('span', class_='relativetime')

            if event_cell and date_cell:
                event_text = event_cell.get_text(strip=True)
                date = date_cell.get('title')

                if 'bounty started' in event_text:
                    bounty_start.append(date)
                elif 'bounty ended' in event_text:
                    bounty_end.append(date)


        if not bounty_start or not bounty_end:
            x = 0
            # print(f"question_id={question_id} missing one or both dates -> start={bounty_start}, end={bounty_end}")

        return {
            'question_id': question_id,
            'bounty_start': bounty_start,
            'bounty_end': bounty_end
        }

    except Exception as e:
        print(f"Error scraping timeline for question_id={question_id}: {e}")
        return None

def process_results(results):
    """
    Save scraped results and update processed IDs.

    Args:
        results (list): List of scraped results.
    """
    if results:
        df = pd.DataFrame(results)
        append_to_parquet(df, RESULTS_PATH)

# Main Execution
if __name__ == "__main__":
    create_data_directory()
    post_ids = get_unprocessed_post_ids()

    if not post_ids:
        print("No unprocessed Post IDs found.")
        exit()

    driver = configure_driver()

    results = []
    with tqdm(total=len(post_ids), desc="Scraping All IDs") as pbar:
        for post_id in post_ids:
            result = scrape_timeline_events(driver, post_id)
            if result:
                results.append(result)

            if len(results) >= WRITE_BATCH_SIZE:
                process_results(results)
                results = []  # Clear the batch
            pbar.update(1)

    # Save any remaining results
    process_results(results)

    driver.quit()