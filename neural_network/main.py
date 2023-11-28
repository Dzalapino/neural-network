

def main():
    import pandas as pd

    # Import the iris dataset
    iris_df = pd.read_csv('resources/iris.csv', index_col=0)
    print(iris_df)


if __name__ == '__main__':
    main()
