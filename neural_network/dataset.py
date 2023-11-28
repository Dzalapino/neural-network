import pandas as pd


class Dataset:
    def __init__(self, path: str, training_ratio=0.8, has_index_col=True, if_shuffle=True):
        # Import the data
        self.df = (pd.read_csv(path, index_col=0) if has_index_col else pd.read_csv(path))
        # Shuffle the data
        if if_shuffle:
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Get the training set size based on given training ratio
        train_set_size = int(training_ratio * len(self.df))

        # Store training and evaluation sets
        self.train_df = self.df.iloc[:train_set_size]
        self.eval_df = self.df.iloc[train_set_size:]
