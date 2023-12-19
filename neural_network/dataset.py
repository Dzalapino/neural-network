"""
Module responsible for the datasets management from the csv file
it also splits the data into training and evaluation sets based on the given ratio
"""
import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, path: str, training_ratio=0.8, has_index_col=True, if_shuffle=True):
        """
        Method for initializing the dataset from the csv file and splitting it
        into training and evaluation sets based on the given ratio
        :param path: path to the csv file
        :param training_ratio: ratio of the training set size to the whole dataset size
        :param has_index_col: if the csv file has index column
        :param if_shuffle: if the dataset should be shuffled
        """
        # Import the data from the csv file
        self.df: pd.DataFrame = (pd.read_csv(path, index_col=0) if has_index_col else pd.read_csv(path))
        # Shuffle the data if needed
        if if_shuffle:
            self.df: pd.DataFrame = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Get the training set size, number of columns and unique label classes
        train_set_size = int(training_ratio * len(self.df))
        n_cols = len(self.df.columns)
        unique_classes = self.df.iloc[:, -1].unique()

        # Get the training set and store training features and labels
        train_df: pd.DataFrame = self.df.iloc[:train_set_size]
        self.train_X = np.array(train_df.iloc[:, :n_cols - 1].to_numpy())
        # Transform the training label samples with one hot encoding
        self.train_y = np.array([[1 if sample == unique else 0 for unique in unique_classes]
                                 for sample in train_df.iloc[:, -1].to_numpy().flatten()])

        # Get the evaluation set and store evaluation features and labels
        eval_df: pd.DataFrame = self.df.iloc[train_set_size:]
        self.eval_X = np.array(eval_df.iloc[:, :n_cols - 1].to_numpy())
        # Transform the evaluation label samples with one hot encoding
        self.eval_y = np.array([[1 if sample == unique else 0 for unique in unique_classes]
                                for sample in eval_df.iloc[:, -1].to_numpy().flatten()])
