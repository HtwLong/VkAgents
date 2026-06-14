from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def crawl_unified_dataset(url, output_filename="unified_dataset.txt"):
    # Initialize the Firefox driver
    driver = webdriver.Firefox()

    try:
        # 1. Navigate to the webpage
        print(f"Navigating to {url}...")
        driver.get(url)
        
        # Wait a moment for the page to load fully
        time.sleep(2)

        # 2. Find the parent element with id "unified_dataset"
        parent_element = driver.find_element(By.ID, "unified_dataset")

        # 3. Find all 'span' tags inside that specific parent element
        spans = parent_element.find_elements(By.TAG_NAME, "span")

        # 4. Extract the text from each span
        # We filter out empty strings to keep the text file clean, 
        # but you can remove the 'if span.text.strip()' part if you want to keep blank lines.
        span_texts = [span.text for span in spans if span.text.strip()]
        count = len(span_texts)

        # 5. Write to a text file
        with open(output_filename, "w", encoding="utf-8") as f:
            # Join all text items with a newline character
            full_content = "\n".join(span_texts)
            f.write(full_content)

        # Output results to console
        print(f"--- Success ---")
        print(f"Found {count} non-empty spans.")
        print(f"All text has been saved to: {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # 6. Close the browser window
        driver.quit()

# --- Usage ---
target_url = "https://vision.semkg.org/explorer.html" 
crawl_unified_dataset(target_url)