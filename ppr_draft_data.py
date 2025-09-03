import pandas as pd
from bs4 import BeautifulSoup
import time
import requests
from thefuzz import process

# --- Selenium Imports ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

import pandas as pd
from bs4 import BeautifulSoup
import time

# --- Selenium Imports ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException


def create_player_df_from_web(url):
    """
    Fetches and parses HTML from the "Sortable Projections" view on ESPN,
    which uses a two-table layout. It handles dynamic content and pagination
    with Selenium, and returns a complete DataFrame.

    This version includes a robust pagination check to ensure the table
    updates before scraping the next page.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    players_data_list = []
    last_first_player_name = None  # To track the first player on the previous page

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        print("Loading initial page...")

        # --- STEP 1: CLICK THE "SORTABLE PROJECTIONS" BUTTON ---
        sortable_button_xpath = "//button[span[text()='Sortable Projections']]"
        sortable_button = wait.until(EC.element_to_be_clickable((By.XPATH, sortable_button_xpath)))
        sortable_button.click()
        print("Clicked 'Sortable Projections' button. Waiting for tables to load...")

        # Wait for the initial table to be visible after the click
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "table.Table--fixed-left")))

        page_count = 1
        while True:
            print(f"Scraping page {page_count}...")

            # --- Wait for table rows to be present before scraping ---
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.Table--fixed-left tbody tr")))
            time.sleep(1) # Extra moment for data to settle

            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')

            # --- STEP 2: LOCATE BOTH TABLES ---
            player_info_table = soup.select_one('table.Table--fixed-left')
            stats_table = soup.select_one('table.Table--fixed-right')

            if not player_info_table or not stats_table:
                print(f"Error: Could not find tables on page {page_count}.")
                break

            player_rows = player_info_table.select('tbody tr.Table__TR')
            stats_rows = stats_table.select('tbody tr.Table__TR')

            if not player_rows:
                print("No player rows found on the page. Ending scrape.")
                break

            # --- Store the first player's name for the pagination check ---
            first_player_on_page = player_rows[0].select_one('a.AnchorLink').get_text(strip=True)
            last_first_player_name = first_player_on_page

            # --- STEP 3: ITERATE THROUGH ROWS AND EXTRACT DATA ---
            for player_row, stats_row in zip(player_rows, stats_rows):
                try:
                    rank = int(player_row.select_one('div.ranking').get_text(strip=True))
                    name = player_row.select_one('a.AnchorLink').get_text(strip=True)
                    position_span = player_row.select_one('span.playerinfo__playerpos')
                    position = position_span.get_text(strip=True) if position_span else 'N/A'
                    team_span = player_row.select_one('span.playerinfo__playerteam')
                    team = team_span.get_text(strip=True) if team_span else 'N/A'
                    points_cell = stats_row.select_one('div.total span')
                    projected_points = float(points_cell.get_text(strip=True)) if points_cell else 0.0

                    players_data_list.append({
                        'Rank': rank,
                        'Name': name,
                        'Team': team,
                        'Position': position,
                        'ProjectedPoints': projected_points
                    })
                except (AttributeError, ValueError, IndexError, TypeError) as e:
                    print(f"Could not parse a row, skipping. Error: {e}")
                    continue

            # --- NEW: ROBUST PAGINATION LOGIC ---
            try:
                next_button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'button.Pagination__Button--next')))
                if next_button.get_attribute('disabled'):
                    print("Last page reached. Scraping complete.")
                    break

                driver.execute_script("arguments[0].click();", next_button)
                page_count += 1

                # --- Verification and Retry Loop ---
                max_retries = 10
                for attempt in range(max_retries):
                    try:
                        # Wait for the first row of the *new* table to be visible
                        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "table.Table--fixed-left tbody tr:first-child")))
                        time.sleep(5) # Allow extra time for JS to update the name
                        
                        # Get the name of the first player on the new page
                        new_first_player_name = driver.find_element(By.CSS_SELECTOR, 'table.Table--fixed-left tbody tr:first-child a.AnchorLink').text
                        
                        if new_first_player_name != last_first_player_name:
                            print(f"Page {page_count} loaded successfully. New first player: {new_first_player_name}")
                            break # Success, exit the retry loop
                        else:
                            # Page did not update, try the back-and-forth maneuver
                            print(f"Page content unchanged. Retrying... (Attempt {attempt + 1}/{max_retries})")
                            
                            # Go back to the previous page
                            prev_button = driver.find_element(By.CSS_SELECTOR, 'button.Pagination__Button--previous')
                            driver.execute_script("arguments[0].click();", prev_button)
                            wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "table.Table--fixed-left tbody tr:first-child")))
                            time.sleep(5)

                            # Go forward to the next page again
                            next_button_retry = driver.find_element(By.CSS_SELECTOR, 'button.Pagination__Button--next')
                            driver.execute_script("arguments[0].click();", next_button_retry)
                        
                        if attempt == max_retries - 1:
                            print(f"Failed to load new page content after {max_retries} attempts. Stopping scrape.")
                            # Set loop condition to false to exit gracefully
                            next_button = None # To break the outer while loop
                            break
                    except TimeoutException:
                        print(f"Timeout waiting for page {page_count} to load on attempt {attempt + 1}.")
                        if attempt == max_retries - 1:
                            next_button = None # To break the outer while loop

                if not next_button: # If we failed all retries, break the main while loop
                    break

            except NoSuchElementException:
                print("No more pages found. Scraping complete.")
                break
            except Exception as e:
                print(f"An unexpected error occurred during pagination: {e}")
                break
    finally:
        driver.quit()

    if not players_data_list:
        print("Warning: No player data was successfully extracted.")
        return pd.DataFrame()

    player_df = pd.DataFrame(players_data_list)
    return player_df

def fetch_adp_data(year=2025, teams=12, ppr=True):
    """
    Fetches Average Draft Position (ADP) data from Fantasy Football Calculator.

    Args:
        year (int): The year for the ADP data.
        teams (int): The number of teams in the league (8, 10, 12, 14).
        ppr (bool): True for PPR leagues, False for standard.

    Returns:
        pandas.DataFrame: A DataFrame containing the ADP data for players,
        or None if the request fails.
    """
    # Determine the format string based on whether it's a PPR league
    format_str = "ppr" if ppr else "standard"

    # Construct the API URL
    url = f"https://fantasyfootballcalculator.com/api/v1/adp/{format_str}?teams={teams}&year={year}"
    

    print(f"Fetching data from: {url}")

    try:
        # Make the GET request to the API
        response = requests.get(url)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        # Parse the JSON response
        data = response.json()
        # The player data is under the 'players' key
        players_list = data.get('players', [])
        print(players_list)

        if not players_list:
            print("No players found in the API response.")
            return None

        # Create a pandas DataFrame from the list of player dictionaries
        df = pd.DataFrame(players_list)
        # Optional: Set the player_id as the index
        df.set_index('player_id', inplace=True)
        return df

    except requests.exceptions.RequestException as e:
        print(f"An error occurred with the network request: {e}")
        return None
    except ValueError as e:
        # Catches JSON decoding errors
        print(f"An error occurred parsing the JSON response: {e}")
        return None


# --- Main execution block ---
if __name__ == '__main__':
    projection_url = "https://fantasy.espn.com/football/players/projections"
    df = create_player_df_from_web(projection_url)
    #df = pd.DataFrame()
    
    if not df.empty:
        print("\n--- Successfully Extracted Player Projections ---")
        print(f"Total players found: {len(df)}")
        print("\n--- First 5 Rows ---")
        print(df.head())
        print("\n--- Last 5 Rows ---")
        print(df.tail())
        print("\n--- DataFrame Info ---")
        df.info()

    # Fetch the data for a 12-team PPR league for the year 2025
    adp_df = fetch_adp_data(year=2025, teams=12, ppr=True)

    # Check if the DataFrame was created successfully
    if adp_df is not None:
        print("\nSuccessfully created DataFrame.")
        print("First 5 rows of the ADP data:")
        print(adp_df.head())
        print("\nDataFrame Info:")
        adp_df.info()

    if not df.empty and adp_df is not None:
        print("\n--- Starting Fuzzy Match and Merge --- 👍")

        # Get the list of names from the ADP dataframe to match against
        adp_names = adp_df['name'].tolist()

        # Define a function to find the best match for a name
        # We set a score_cutoff of 85 to avoid bad matches
        def get_best_match(name, choices, score_cutoff=85):
            match = process.extractOne(name, choices)
            if match and match[1] >= score_cutoff:
                return match[0]  # Return the matched name
            return None

        # Apply the function to the 'Name' column of the ESPN dataframe
        # This creates a new column with the best-matched name from adp_df
        print("Finding best name matches...")
        df['matched_name'] = df['Name'].apply(get_best_match, args=(adp_names,))

        # Now, merge the two DataFrames using the matched name
        # We use a left merge to keep all players from the original ESPN df
        merged_df = pd.merge(
            df,
            adp_df,
            left_on='matched_name',
            right_on='name',
            how='left'
        )

        # Clean up and display the results
        # We can drop the redundant name columns if desired
        final_df = merged_df.drop(columns=['matched_name', 'name'])

        print("\n--- Merge Complete ---")
        print("Showing a sample of the merged data (only successfully matched rows):")
        # Display rows where the merge was successful
        print(final_df[final_df['adp'].notna()].head())

        print("\n--- Final Merged DataFrame Info ---")
        final_df.info()
        final_df.to_csv("data4.csv")
