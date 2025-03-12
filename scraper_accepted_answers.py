import os
import re
import asyncio
import pandas as pd
import duckdb
from tqdm.asyncio import tqdm_asyncio
from bs4 import BeautifulSoup
import aiohttp
import shlex
from fastparquet import write
import numpy as np
from scipy import stats
import time
import random

# Constants
DATA_DIR = "./Data"
RESULTS_PATH = os.path.join(DATA_DIR, "accepted_answer_timeline_results.parquet")
QUESTIONS_PATH = os.path.join(DATA_DIR, "posts_questions.parquet")
ANSWERS_PATH = os.path.join(DATA_DIR, "posts_answers.parquet")
WRITE_BATCH_SIZE = 1000
MAX_CONCURRENT_REQUESTS = 20  # Adjust based on your system and rate limits

# Dynamic rate limiting parameters
BASE_MIN_DELAY = 0.01  # Minimum base delay
BASE_MAX_DELAY = 0.05  # Maximum base delay


class SophisticatedRateLimiter:
    """
    A rate limiter that uses layered probabilistic distributions
    to create human-like, unpredictable request patterns.
    """

    def __init__(self):
        # Parameters for the meta-distribution that determines lambda values
        self.update_meta_parameters()
        # Initialize the current lambda for the Poisson distribution
        self.update_lambda()
        # Timestamp for when to update parameters
        self.next_param_update = time.time() + self._get_update_interval()
        # Keep track of request times for self-monitoring
        self.request_times = []

    def update_meta_parameters(self):
        """Update the parameters used for generating lambda values"""
        # Mix of normal distributions for more natural patterns
        self.mu1 = random.uniform(0.5, 2.0)
        self.sigma1 = random.uniform(0.1, 0.5)
        self.mu2 = random.uniform(1.5, 3.0)
        self.sigma2 = random.uniform(0.2, 0.7)
        # Weight between the two distributions
        self.weight = random.uniform(0.3, 0.7)

    def update_lambda(self):
        """Generate a new lambda value for the Poisson distribution"""
        # Mix two normal distributions for the lambda parameter
        if random.random() < self.weight:
            lambda_val = max(0.5, np.random.normal(self.mu1, self.sigma1))
        else:
            lambda_val = max(0.5, np.random.normal(self.mu2, self.sigma2))

        # Create a new Poisson distribution with this lambda
        self.poisson = stats.poisson(lambda_val)

    def _get_update_interval(self):
        """Determine when to update parameters (also random)"""
        # Update parameters every 10-30 seconds
        return random.uniform(10, 30)

    async def get_delay(self):
        """Get the next delay value using our layered distributions"""
        current_time = time.time()

        # Check if we need to update parameters
        if current_time >= self.next_param_update:
            # Occasionally update meta-parameters (with 20% chance)
            if random.random() < 0.2:
                self.update_meta_parameters()
            # Always update lambda
            self.update_lambda()
            # Set next update time
            self.next_param_update = current_time + self._get_update_interval()

        # Get a random value from Poisson distribution
        poisson_value = self.poisson.rvs()

        # Scale to our desired delay range and add tiny random variation
        delay = BASE_MIN_DELAY + (poisson_value / 10.0) * (BASE_MAX_DELAY - BASE_MIN_DELAY)

        # Add small random noise for even more unpredictability
        delay += random.uniform(-0.005, 0.005)

        # Ensure delay stays within reasonable bounds
        delay = max(BASE_MIN_DELAY, min(BASE_MAX_DELAY * 2, delay))

        # Occasionally add a longer pause (simulating human behavior)
        if random.random() < 0.01:  # 1% chance
            delay += random.uniform(0.5, 2.0)

        # Log this request time for self-monitoring
        self.request_times.append(current_time)

        # Clean up old request times
        self._clean_history()

        return delay

    def _clean_history(self):
        """Remove request times older than 5 minutes to conserve memory"""
        cutoff = time.time() - 300  # 5 minutes
        self.request_times = [t for t in self.request_times if t >= cutoff]

    def get_stats(self):
        """Return statistics about recent request patterns"""
        if len(self.request_times) < 2:
            return {"count": len(self.request_times), "rate": 0}

        # Calculate recent request rate (requests per second)
        recent_times = [t for t in self.request_times if t >= time.time() - 60]
        if len(recent_times) >= 2:
            time_span = max(recent_times) - min(recent_times)
            rate = len(recent_times) / time_span if time_span > 0 else 0
        else:
            rate = 0

        return {
            "count": len(self.request_times),
            "recent_count": len(recent_times),
            "rate": rate,
            "current_lambda": self.poisson.mean()
        }


def parse_curl_command(curl_command):
    """
    Parse a curl command to extract headers and cookies
    """
    # Split the command into tokens while preserving quoted strings
    tokens = shlex.split(curl_command)

    url = None
    headers = {}
    cookies = {}

    i = 1  # Skip 'curl' at index 0
    while i < len(tokens):
        if tokens[i] == '-H' or tokens[i] == '--header':
            if i + 1 < len(tokens):
                header_line = tokens[i + 1]
                if ':' in header_line:
                    key, value = header_line.split(':', 1)
                    headers[key.strip()] = value.strip()
                i += 2
            else:
                i += 1
        elif tokens[i] == '-b' or tokens[i] == '--cookie':
            if i + 1 < len(tokens):
                cookie_string = tokens[i + 1]
                cookie_pairs = re.split('; |;', cookie_string)
                for pair in cookie_pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        cookies[key.strip()] = value.strip()
                i += 2
            else:
                i += 1
        elif tokens[i].startswith('http'):
            url = tokens[i].strip("'")
            i += 1
        else:
            i += 1

    return url, headers, cookies


def create_data_directory():
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def get_unprocessed_post_ids():
    """Get list of question IDs that need processing."""
    con = duckdb.connect()

    # Create views
    con.execute(f"""
        CREATE TEMPORARY VIEW questions AS
        SELECT
            Id AS question_id,
            AcceptedAnswerId AS accepted_answer_id,
            CAST(CreationDate AS TIMESTAMP) AS creation_date
        FROM '{QUESTIONS_PATH}';
    """)

    con.execute(f"""
        CREATE TEMPORARY VIEW answers AS
        SELECT
            Id AS answer_id,
            ParentId AS parent_question_id,
            CAST(CreationDate AS TIMESTAMP) AS creation_date
        FROM '{ANSWERS_PATH}';
    """)

    # Check if results file exists
    if os.path.exists(RESULTS_PATH):
        query = f"""
            SELECT DISTINCT q.question_id
            FROM questions q
            JOIN answers a ON q.accepted_answer_id = a.answer_id
            WHERE q.accepted_answer_id IS NOT NULL
            AND q.question_id NOT IN (
                SELECT DISTINCT r.question_id
                FROM parquet_scan('{RESULTS_PATH}') r
            )
        """
    else:
        query = """
            SELECT DISTINCT q.question_id
            FROM questions q
            JOIN answers a ON q.accepted_answer_id = a.answer_id
            WHERE q.accepted_answer_id IS NOT NULL
        """

    post_ids_data = con.execute(query).fetchall()
    con.close()
    return [row[0] for row in post_ids_data]


def append_to_parquet(dataframe, path):
    """Append to existing Parquet file or create a new one."""
    if os.path.exists(path):
        write(path, dataframe, append=True)
    else:
        write(path, dataframe)


async def fetch_timeline(session, question_id, semaphore, rate_limiter):
    """Fetch timeline for a specific question."""
    url = f'https://stackoverflow.com/posts/{question_id}/timeline'

    async with semaphore:
        try:
            # Get delay from our sophisticated rate limiter
            delay = await rate_limiter.get_delay()
            await asyncio.sleep(delay)

            async with session.get(url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')

                    # Look for vote events (acceptance is considered a vote)
                    vote_rows = soup.find_all(
                        'tr',
                        attrs={
                            'data-eventtype': 'vote',
                            'class': lambda x: x and 'datehash' in x
                        }
                    )

                    accept_timestamp = None
                    for row in vote_rows:
                        event_cell = row.find('td', class_='wmn1')
                        date_cell = row.find('span', class_='relativetime')

                        if event_cell and date_cell:
                            event_text = event_cell.get_text(strip=True)
                            date = date_cell.get('title')

                            if 'accept' in event_text.lower():
                                accept_timestamp = date
                                break
                    # print(f"Found for question_id {question_id} the accepted timestamp: {accept_timestamp}")
                    return {
                        'question_id': question_id,
                        'accept_timestamp': accept_timestamp
                    }
                else:
                    print(f"Error {response.status} for question_id={question_id}")

                    # If we hit rate limiting, pause longer
                    if response.status in (429, 503):
                        print("Rate limit detected, pausing...")
                        await asyncio.sleep(random.uniform(5, 10))

                    return None
        except Exception as e:
            print(f"Exception for question_id={question_id}: {e}")
            return None


async def main():
    # Parse the curl command to get headers and cookies
    curl_command = """curl 'https://stackoverflow.com/posts/76353940/timeline' \
      -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
      -H 'accept-language: en-DE,en;q=0.9,de-DE;q=0.8,de;q=0.7,en-US;q=0.6' \
      -b 'prov=6fb614ed-8d51-4787-be72-c9863376e0a8; OptanonAlertBoxClosed=2024-12-01T09:43:15.703Z; eupubconsent-v2=CQI8_lgQI8_lgAcABBENBSF4AP_AAEPAACiQKiQBgAGAAgAHQAaABecFQwVEAvOAGACAAdAC8wAA.f_gACHgAAAAA; *so*tgt=1ac0e1f6-a9a7-4634-8d80-6248ab7b92d6; usr=p=%5b160%7c%3bBountyEndingSoon%3bBounty%3b%5d; *ga*S812YQPLT2=GS1.1.1739043810.4.1.1739043961.0.0.0; *ga*WCZ03SZFCQ=deleted; __cflb=0H28vFHtoAR1ohjxFgDZBwZ5H5NWURXoth2kCfmctov; *cfuvid=xDrHvto6DZzx7fu40VlzKLNSl0G5KWyZe1*_HOrwCaU-1741798062536-0.0.1.1-604800000; *gid=GA1.2.1216424359.1741798066; acct=t=Jnopa4jnOFsxYRy5Uez3%2fUFzhP9IC4IX&s=gOgc%2f23Dl4kl3O0fkibiTaouFnxbE%2bqX; *_cf_bm=kViScPehU53GvzkWPkj.W2pI_VXxcPGxIPQ5h7dWZzo-1741798971-1.0.1.1-2yZRoNs3xcRcYn4g.7t5S4kvhYPkLfkwyuUiwaDJqHEtHQTxZnYlFxUHwkt0YdZS6dbGtk.uxTndoIZ32vyCSYiJ8D8TX.OFuHIPDV2BwwE; cf_clearance=ZYeuipTXFbrGx.KUXOTI_QYzoLh4PLkWRVsno8LkOAc-1741798973-1.2.1.1-Vm9q6zIlL6olZYe9mzmrjD5KNExmQrC4h4R9DqjA9DU9jclpLsciMptQtZhzyHtuDX_slqXwL1PR32Nu7YF3tIzHATqkEfxChIzkYrju6M1Bz7GUf8PshaYxJgHY6CEfIO8Pjj4mjDSp5EvKM2TC_sGfX6_m7v5yGOSNhYCtEv5AiQGMAtGCqCsxgTuiki67CSsIYpvuhei.iGWdTIjRHyERL7xUtE1cDwO8iyyDED8g4goi636djuZ5BZRWmwQvWu6pog1ciUMSCUxiNKQJkmx0uSsmNLvYv6XIyQUnb.xheXyg7SO7OvPOgWyPfUBrtr.WmESJT8p1oIuUVL8.avc9.a.S5vHeklqm2.emBqo; *gat=1; *ga_WCZ03SZFCQ=GS1.1.1741798066.11.1.1741798974.59.0.0; _ga=GA1.1.1013657177.1733128174; OptanonConsent=isGpcEnabled=0&datestamp=Wed+Mar+12+2025+18%3A02%3A54+GMT%2B0100+(Central+European+Standard+Time)&version=202411.2.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=f8fbb4a0-22e6-4e01-a766-e1a12e3d87a7&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0003%3A1%2CC0002%3A1%2CC0004%3A1&intType=1&geolocation=DE%3BNW&AwaitingReconsent=false&genVendors=V3%3A0%2C' \
      -H 'priority: u=0, i' \
      -H 'sec-ch-ua: "Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"' \
      -H 'sec-ch-ua-mobile: ?0' \
      -H 'sec-ch-ua-platform: "Windows"' \
      -H 'sec-fetch-dest: document' \
      -H 'sec-fetch-mode: navigate' \
      -H 'sec-fetch-site: none' \
      -H 'sec-fetch-user: ?1' \
      -H 'upgrade-insecure-requests: 1' \
      -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'"""

    _, headers, cookies = parse_curl_command(curl_command)

    create_data_directory()
    post_ids = get_unprocessed_post_ids()

    if not post_ids:
        print("No unprocessed questions with accepted answers found.")
        return

    # For testing/development, use a smaller set
    # post_ids = post_ids[:1000]  # Uncomment for testing

    print(f"Found {len(post_ids)} unprocessed questions")

    # Initialize our sophisticated rate limiter
    rate_limiter = SophisticatedRateLimiter()

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Set up connection pooling with rotating user agents
    conn = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS, force_close=False)
    timeout = aiohttp.ClientTimeout(total=30)

    # Add ability to rotate user agents
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Edge/134.0.0.0"
    ]

    # Start a background task to periodically report stats
    async def report_stats():
        while True:
            await asyncio.sleep(60)  # Report every minute
            stats = rate_limiter.get_stats()
            print(f"Rate stats: {stats}")

    stats_task = asyncio.create_task(report_stats())

    try:
        async with aiohttp.ClientSession(
                headers=headers,
                cookies=cookies,
                connector=conn,
                timeout=timeout
        ) as session:
            tasks = []

            # Create tasks for all question IDs
            for question_id in post_ids:
                # Occasionally rotate user agent
                if random.random() < 0.1:  # 10% chance to change user agent
                    new_ua = random.choice(user_agents)
                    session.headers['User-Agent'] = new_ua

                task = fetch_timeline(session, question_id, semaphore, rate_limiter)
                tasks.append(task)

            # Process tasks in batches with progress tracking
            for i in range(0, len(tasks), WRITE_BATCH_SIZE):
                batch = tasks[i:i + WRITE_BATCH_SIZE]

                # Add some natural variability to batch size
                actual_batch_size = min(
                    len(batch),
                    WRITE_BATCH_SIZE + random.randint(-100, 100)
                )
                actual_batch = batch[:actual_batch_size]

                batch_results = await tqdm_asyncio.gather(
                    *actual_batch,
                    desc=f"Processing batch {i // WRITE_BATCH_SIZE + 1}/{len(tasks) // WRITE_BATCH_SIZE + 1}"
                )

                # Filter out None results
                valid_results = [r for r in batch_results if r is not None]
                if valid_results:
                    # Save batch results
                    results_df = pd.DataFrame(valid_results)
                    append_to_parquet(results_df, RESULTS_PATH)
                    print(f"Saved batch of {len(valid_results)} results")

                # Instead of fixed delays between batches, use variable delays
                # that follow a more natural pattern
                if i + WRITE_BATCH_SIZE < len(tasks):  # If not the last batch
                    # Sometimes take longer breaks between batches
                    if random.random() < 0.2:  # 20% chance
                        pause_time = random.uniform(2, 5)
                        print(f"Taking a {pause_time:.1f}s pause between batches...")
                        await asyncio.sleep(pause_time)
    finally:
        # Clean up the stats reporting task
        stats_task.cancel()
        try:
            await stats_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())