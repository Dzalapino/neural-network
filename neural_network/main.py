

def main():
    from neural_network.dataset import Dataset

    dataset = Dataset('resources/iris.csv', 0.8)
    print(dataset.df)
    print(dataset.train_df)
    print(dataset.eval_df)


if __name__ == '__main__':
    main()
