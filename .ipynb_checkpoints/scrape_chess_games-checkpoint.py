import datetime
import time
import os  
from pathlib import Path
import pandas as pd
from selenium import webdriver  
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# load environment variables from a .env file in parent directory
from dotenv import load_dotenv
load_dotenv()

now = datetime.datetime.now()

# save your username and password in a .env file in the parent directory of your workspace
CHESSCOM_USER = os.getenv("CHESSCOM_USER") 
PASSWORD = os.getenv("PASSWORD")
order = ["asc", "desc"]

for i, o in enumerate(order):
    GAMES_URL = "https://www.chess.com/games/archive?gameOwner=other_game&username=" + CHESSCOM_USER \
        + "&gameType=live&gameResult=&opponent=&opening=&color=&gameTourTeam=&" + \
            "timeSort=" + o + "&rated=rated&startDate%5Bdate%5D=12%2F01%2F2020&endDate%5Bdate%5D=" + \
                str(now.month) + "%2F" + str(now.day) + "%2F" + str(now.year) + \
                    "&ratingFrom=&ratingTo=&page="
    LOGIN_URL = "https://www.chess.com/login"

    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options = options)
    driver.get(LOGIN_URL)
    driver.find_element(By.ID, "username").send_keys(CHESSCOM_USER)
    driver.find_element(By.ID, "password").send_keys(PASSWORD)
    driver.find_element(By.ID, "login").click()
    time.sleep(5)

    tables = []
    game_links = []

    for page_number in range(1,100):
        driver.get(GAMES_URL + str(page_number))
        time.sleep(15)
        tables.append(
            pd.read_html(
                driver.page_source, 
                attrs={'class':'table-component table-hover archive-games-table'}
            )[0]
        )
        
        table_user_cells = driver.find_elements(By.CLASS_NAME, 'archive-games-user-cell')
        for cell in table_user_cells:
            link = cell.find_element(By.TAG_NAME, 'a')
            game_links.append(link.get_attribute('href'))

    driver.close()

    games = pd.concat(tables)
    game_links = pd.Series(game_links)
    games['game_link'] = game_links

    if i == 0:
        filepath = Path('early_career_games.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(filepath)
    elif i == 1:
        filepath = Path('late_career_games.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(filepath)