# My-Chess-Progression

I scrape data from my games played on chess.com. I use `selenium` to scrape the data from the library page of my profile. 
Anyone with a chess.com account can use the scripts to download their own data. One important note is the `review_games.py` script, which first reviews every game I have
played before then executing the `scrape_chess_games.py` file. I am a gold chess.com member and thus have unlimited game reviews. The only piece of information I obtain
from reviewing the games is my accuracy and my opponent's accuracy, which I also wanted to analyse. 

The structure of the project is as follows:

- My-Chess-Progression
  - data
    - ML_processed.csv
    - early_career_games.csv
    - games_cleaned.csv
    - late_career_games.csv
  - modelling
    - dataset.py
    - models.py
  - scripts
    - scrape_chess_games.py
    - review_games.py
  - .gitignore
  - Data_Cleaning.ipynb
  - Insights.ipynb
  - ML_Preprocessing.ipynb
  - WinLoss_Modelling.ipynb
  - requirements.txt

The `data` subfolder contains all of the pre-processed as well as the processed data after cleaning and preparing for modelling.
The `modelling` subfolder contains the `models.py` file which contains the classes and methods for implementing the models, as well as the `dataset.py` file containing the methods
for preparing the datset for modelling.
The `scripts` subfolder contains the scripts for scraping the data from chess.com.
The iPython notebooks:
  - Data_Cleaning.ipynb is for cleaning the raw data from chess.com
  - Insights.ipynb is for analysing the data and gathering insights about my play
  - ML_preprocessing.ipynb is for preprocessing the data for the ML models
  - WinLoss_Modelling.ipynb is for implementing the models and making predictions.







