import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

from dataset import dataset

numerical_vars = ['moves', 'my_rating', 'opponent_rating', 'my_accuracy', 'opponent_accuracy', 'rating_difference']
ohe_vars = ['time_format_Blitz', 'time_format_Rapid', 'time_format_Bullet', 'colour_Black', 'colour_White']
outcome_var = ['win']
    
class Model:
    '''We will be using the methods of this class for our different models.'''

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.available_hyperparams = None
        self.initial_model = None
        self.param_grid = None
        self.n_folds = None
        self.best_estimator = None
        self.y_pred = None
        self.name = None
    
    def get_hyperparams(self):
        '''Get available hyperparameters for model.'''

        # return available hyperparameters for model
        self.available_hyperparams = list(self.initial_model.get_params().keys())
        return self.available_hyperparams

    def set_param_grid(self, **hyperparams):
        '''Define (hyperparameter grid) possible values for hyperparams for model.'''

        self.param_grid = hyperparams
        return self.param_grid

    def train_model(self, n_folds: int = 5):
        '''We will use grid search with 5-fold (default) cross-validation to find the best parameters for our model.'''

        # set number of folds for cross-validation
        self.n_folds = n_folds

        # initialise grid search with n-fold cross-validation for logistic regression and fit to training set
        print(f'Using Grid search to find best parameters for {self.name} model...')
        grid = GridSearchCV(self.initial_model,
                            param_grid = self.param_grid,
                            cv = self.n_folds,
                            scoring = 'accuracy',
                            verbose = 1)
        grid.fit(self.X_train, self.y_train)

        # print best parameters and highest accuracy found
        print('Best parameters found: ', grid.best_params_)
        print(f'Highest accuracy found:  {np.round(grid.best_score_*100, 2)}%')

        # choose best estimator from grid search
        print('Choosing best estimator from grid search...')
        self.best_estimator = grid.best_estimator_

        return self.best_estimator
    
    def evaluate_model(self):
        '''Evaluate model on test set. User can add more metrics if needed.'''

        # predict on test set
        self.y_pred = self.best_estimator.predict(self.X_test)

        # calculate evaluation metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred, pos_label = 1)

        print(f'The accuracy of our {self.name} model is: {np.round(accuracy, 3)*100}%.')
        print(f'The recall of our {self.name} model is: {np.round(recall, 3)*100}%.')

    def make_prediction(self, X: dict(), dataset: pd.DataFrame = dataset):
        '''Make prediction on a single row of new data. Tells user probability of outcome as well.
           Takes dictionary of input data as argument with columns ordered as in dataset.py.'''

        # Check data types and values
    
        # Check if moves, my_rating, opponent_rating, rating_difference are integers
        integer_vars = ['moves', 'my_rating', 'opponent_rating', 'rating_difference']
        for var in integer_vars:
            if not isinstance(X[var], int):
                raise ValueError(f"'{var}' should be an integer.")
    
        # Check if my_accuracy, opponent_accuracy are floats
        float_vars = ['my_accuracy', 'opponent_accuracy']
        for var in float_vars:
            if not isinstance(X[var], float):
                raise ValueError(f"'{var}' should be a float.")
        
        # Check if the rest of the columns are only either 1s or 0s
        boolean_vars = ['time_format_Bullet', 'time_format_Blitz', 'time_format_Rapid', 'colour_White', 'colour_Black']
        for var in boolean_vars:
            if X[var] not in [0, 1]:
                raise ValueError(f"'{var}' should be either 0 or 1.")
    
        # Check additional conditions

        # Check if my_rating - opponent_rating equals rating_difference
        if X['my_rating'] - X['opponent_rating'] != X['rating_difference']:
            raise ValueError("The difference between 'my_rating' and 'opponent_rating' should be equal to 'rating_difference'.")

        # Check if only one of time_format_Bullet, time_format_Blitz, time_format_Rapid has value 1
        time_formats = [X['time_format_Bullet'], X['time_format_Blitz'], X['time_format_Rapid']]
        if sum(time_formats) != 1:
            raise ValueError("Only one of 'time_format_Bullet', 'time_format_Blitz', 'time_format_Rapid' should have a value of 1.")

        # Check if only one of colour_White, colour_Black has value 1
        colors = [X['colour_White'], X['colour_Black']]
        if sum(colors) != 1:
            raise ValueError("Only one of 'colour_White' and 'colour_Black' should have a value of 1.")

        # convert input data to dataframe
        X = pd.DataFrame(X, index = [0])

        # standardise numerical variables
        for col in numerical_vars:
            X[col] = X[col].apply(lambda x: (x - dataset[col].mean()) / dataset[col].std())

        # make prediction
        prediction = self.best_estimator.predict(X)

        # get probability of prediction
        probabilities = self.best_estimator.predict_proba(X)
        probability = (probabilities[0][1] if prediction == 1 else probabilities[0][0])

        print(f"The predicted outcome is a {'win' if prediction == 1 else 'loss'}. This is predicted with a probability of {np.around(probability, decimals = 2)}.")

    def plot_confusion_matrix(self):
        '''Plot confusion matrix for model.'''

        cm = confusion_matrix(self.y_test, self.y_pred)
        labels = self.best_estimator.classes_

        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax = ax, fmt = 'd')

        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix for {self.name} model')
        ax.yaxis.set_tick_params(rotation=360)

        ax.xaxis.set_ticklabels(labels) 
        ax.yaxis.set_ticklabels(labels)

class LogReg(Model):
    '''We will be using logistic regression from scikit-learn to predict the outcome of a chess game.'''
    
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.initial_model = None
        self.name = 'Logistic Regression'

    def initialise_model(self, random_state: int = 1):
        '''Initialise logistic regression model.'''

        # initialise empty logistic regression model, choose random_state for reproducibility
        print(f'Initialising {self.name} model...')
        self.initial_model = LogisticRegression(random_state = random_state)

        return self.initial_model
    
    def get_coefficients(self):
        '''Get feature coefficients for logistic regression model.'''

        # get coefficients for logistic regression model
        print('Getting coefficients for logistic regression model...')
        coefficients = pd.DataFrame(self.best_estimator.coef_, columns = self.X_train.columns)

        return coefficients

class GradientBoostedModel(Model):
    '''We will be using the xgboost library to make a gradient boosted model with decision trees
    as base learners to predict the outcome of a chess game.'''

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.initial_model = None
        self.name = 'Gradient Boosted'

    def initialise_model(self, random_state: int = 1):
        '''Initialise gradient boosted model.'''

        # initialise empty logistic regression model, choose random_state for reproducibility
        print(f'Initialising {self.name} model...')
        self.initial_model = xgb.XGBClassifier(objective = 'binary:logistic', random_state = random_state)

        return self.initial_model