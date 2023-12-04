import pickle
import torch


def loadFiles(file):
    with open(file, "rb") as file:
        data = pickle.load(file)
    print("The size of the dataset is:", len(data))
    return data


def separateData(data):
    X = data[:, 0]
    y = data[:, 1]

    return X, y






if __name__ == "__main__":
    data = loadFiles("english-german-both.pkl")
    X, y = separateData(data)
