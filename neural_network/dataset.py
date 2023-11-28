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

        n_cols = len(self.df.columns)

        # Get the training set and store training features and labels
        train_df: pd.DataFrame = self.df.iloc[:train_set_size]
        self.train_X = train_df.iloc[:, :n_cols-1].to_numpy()
        self.train_y = train_df.iloc[:, n_cols-1:].to_numpy()

        # Get the evaluation set and store evaluation features and labels
        eval_df: pd.DataFrame = self.df.iloc[train_set_size:]
        self.eval_X = eval_df.iloc[:, :n_cols-1].to_numpy()
        self.eval_y = eval_df.iloc[:, n_cols-1:].to_numpy()
