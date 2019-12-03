import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch

def remove_missing(full_data):
    full_size = full_data.shape[0]
    print('Dataset size Before pruning: ', full_size)
    for data in [full_data]:
        for i in full_data:
            data[i].replace('nan', np.nan, inplace=True)
        data.dropna(inplace=True)
    real_size = full_data.shape[0]
    print('Dataset size after pruning: ', real_size)
    print('We eliminated ', (full_size-real_size), ' datapoints')


def replace_categorical(full_data):
    cat_data = full_data.select_dtypes(include=['object']).copy()
    other_data = full_data.select_dtypes(include=['int']).copy()
    print(cat_data.shape)
    print(other_data.shape)
    newcat_data = pd.get_dummies(cat_data, columns=[
        "Workclass", "Education", "Country", "Relationship",
        "Martial Status", "Occupation", "Relationship",
        "Race", "Sex"
    ])
    return pd.concat([other_data, newcat_data], axis=1)


def separate_label(full_data):
    full_labels = full_data['Target'].copy()
    full_data = full_data.drop(['Target'], axis=1)
    label_encoder = LabelEncoder()
    full_labels = label_encoder.fit_transform(full_labels)
    return full_data, full_labels


def load_adult():

    columns = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status", \
              "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"]

    types = {0: int, 1: str, 2: int, 3: str, 4: int, 5: str, 6: str, 7: str, 8: str, 9: str, 10: int,
                                    11: int, 12: int, 13: str, 14: str}
    full_data = pd.read_csv(
        "data/adult/adult.csv",
        names=columns,
        sep=r'\s*,\s*',
        engine='python', skiprows=1,
        na_values="?",
        dtype=types)

    remove_missing(full_data)
    full_data, full_label = separate_label(full_data)
    full_data = replace_categorical(full_data)
    full_data.head()

    train_num = 30000

    train_data  = full_data.iloc[:train_num]
    train_label = full_label[:train_num]

    test_data   = full_data.iloc[train_num:]
    test_label  = full_label[train_num:]

    return torch.from_numpy(train_data.reset_index().values[:, 1:]),\
           torch.from_numpy(train_label.reshape(-1, 1)),\
           torch.from_numpy(test_data.reset_index().values[:, 1:]),\
           torch.from_numpy(test_label.reshape(-1, 1))


def load_mnist():
    train_data = np.load('data/mnist/image_train.npy').reshape(-1, 784)
    train_label = np.load('data/mnist/label_train.npy')
    train_color = np.load('data/mnist/color_train.npy')

    test_data = np.load('data/mnist/image_test.npy').reshape(-1, 784)
    test_label = np.load('data/mnist/label_test.npy')
    test_color = np.load('data/mnist/color_test.npy')

    # test_data
    test_data = torch.stack([torch.Tensor(i).double() for i in test_data])
    test_color = torch.from_numpy(test_color).unsqueeze(1)
    test_data = torch.cat((test_data, test_color.double()), dim=1)
    test_label = torch.from_numpy(test_label == 2).double().unsqueeze(1)

    # train_data
    train_data = torch.stack([torch.Tensor(i).double() for i in train_data])
    train_color = torch.from_numpy(train_color).unsqueeze(1)
    train_data = torch.cat((train_data, train_color.double()), dim=1)
    train_label = torch.from_numpy(train_label == 2).double().unsqueeze(1)
    return train_data, train_label, test_data, test_label


