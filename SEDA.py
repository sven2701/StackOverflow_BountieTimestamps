import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = uc.Chrome()
driver.get("https://data.stackexchange.com/")
input("Log in manually then press Enter to continue...")

offset = 0
step = 50000
current_href = ""  # initially no CSV download link

base_url = "https://data.stackexchange.com/stackoverflow/csv/"

while True:
    lower = offset + 1
    upper = offset + step
    query = f""";
WITH OrderedPosts AS (
    SELECT p.AcceptedAnswerId,
           p.Id AS QuestionId,
           ROW_NUMBER() OVER (ORDER BY p.Id) AS rn
    FROM Posts p
    JOIN Users u ON p.OwnerUserId = u.Id
    WHERE p.PostTypeId = 1
      AND p.AcceptedAnswerId IS NOT NULL
)
SELECT AcceptedAnswerId,
       QuestionId
FROM OrderedPosts
WHERE rn BETWEEN {lower} AND {upper};
"""
    # Set the query in CodeMirror and click "Run Query"
    driver.execute_script(
        "document.querySelector('.CodeMirror').CodeMirror.setValue(arguments[0]);",
        query
    )
    driver.find_element(By.ID, "submit-query").click()

    # Wait for the messages area to update
    msg_elem = WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div[3]/div[4]/pre"))
    )
    if "(0 row(s) returned)" in msg_elem.text:
        print("No more results; exiting.")
        break

    # Wait for the CSV download button to be clickable
    csv_btn = WebDriverWait(driver, 60).until(
        EC.element_to_be_clickable((By.ID, "resultSetsButton"))
    )

    # For the first iteration, simply record the CSV href.
    # For subsequent iterations, wait until the CSV href is exactly (old numeric id + 1)
    if not current_href:
        new_href = csv_btn.get_attribute("href")
    else:
        try:
            old_id = int(current_href.split("/")[-1])
        except ValueError:
            old_id = 0
        expected_href = base_url + str(old_id + 1)
        new_href = WebDriverWait(driver, 60).until(
            lambda d: d.find_element(By.ID, "resultSetsButton").get_attribute("href") == expected_href and expected_href
        )

    print("Downloading CSV from:", new_href)
    csv_btn.click()

    # Update the current CSV href and wait a moment for download to start/finish
    current_href = new_href
    time.sleep(5)

    offset += step
    time.sleep(2)