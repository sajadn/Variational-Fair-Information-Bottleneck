import os
import numpy as np
import sys
import torch
import torch.nn as nn
from preprocess import load_adult, load_mnist
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from model import reparam, negative_log_bernoulli

d = {'test acc label': [],
     'test acc sens':  []}

model_name = sys.argv[1]
dataset_name = sys.argv[2]

directory = 'trained_models/' + model_name + '/' + dataset_name + '/'

file_num = 0
for filename in os.listdir(directory):
    file_path = directory + filename + '/evaluation.txt'
    print(file_path)
    with open(file_path) as f:
        for line in f:
            (key, val) = line.split(':')
            d[key].append(float(val))



    file_num += 1

for key in d.keys():
    d[key] = np.array(d[key])

with open('trained_models/'+directory.replace('/', '-')[15:-1] + '.txt', 'w') as file:
    file.write('Average of {} runs\n'.format(file_num))
    for key in d.keys():
        file.write(key + ' mean: ' + str(np.mean(d[key])) + ' std: ' + str(np.std(d[key])) +'\n')
    file.close()

# test_epochs = 200
# sensitive_attr = -1 if dataset_name=='mnist' else 0
# latent_size = 50
# lr = 0.0002
#
# train_data, train_label, test_data, test_label = load_adult()
# threshold = [38.64, 189664.13, 10.07, 1079.06, 87.502314, 40.422382]
# for i in range(len(threshold)):
#     train_data[:, i] = (train_data[:, i] > threshold[i]).double()
#     test_data[:, i] = (test_data[:, i] > threshold[i]).double()
#
# train_dataset = data_utils.TensorDataset(train_data, train_label)
# test_dataset = data_utils.TensorDataset(test_data, test_label)
#
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=20000, shuffle=True)
#
# model = torch.load(directory + filename + '/model')
# model.eval()
#
# model.eval()
# sigmoid = torch.nn.Sigmoid()
# test_classifier = nn.Sequential(nn.Linear(50, 1)).double().cuda()
#
# optimizer_label = torch.optim.Adam(test_classifier.parameters(), lr=lr)
#
#
#
# for epoch_number in range(test_epochs):
#     print('epoch number:', epoch_number)
#     for (data, label) in train_loader:
#         label = label.double().cuda()
#         data = data.double().cuda()
#         label_sens = data[:, sensitive_attr].unsqueeze(1)
#         # print('label sens shape', label_sens.shape)
#
#         q_z_mean, q_z_log_sigma = model.encoder(data)
#         q_z_mean = q_z_mean.detach()
#         q_z_log_sigma = q_z_log_sigma.detach()
#         z = reparam(q_z_mean, q_z_log_sigma)
#
#         pred_label = test_classifier(z)
#
#         label_loss = negative_log_bernoulli(label, pred_label)
#
#         optimizer_label.zero_grad()
#
#         # backward
#         label_loss.backward()
#
#         # optimizers step
#         optimizer_label.step()
#
#
#     acc_label = torch.mean(((sigmoid(pred_label).detach() > 0.5).double() == label).double())
#
#
#
#     print('last train batch acc_label:', acc_label)
#
# for (data, label) in test_loader:
#     data = data.double().cuda()
#     label = label.double().cuda()
#
#     label_sens = data[:, sensitive_attr].unsqueeze(1)
#
#     q_z_mean, q_z_log_sigma = model.encoder(data)
#     q_z_mean = q_z_mean.detach()
#     q_z_log_sigma = q_z_log_sigma.detach()
#     z = reparam(q_z_mean, q_z_log_sigma)
#
#     # Prediction
#     pred_label = sigmoid(test_classifier(z))
#
#     disct1 = torch.mean(pred_label[label_sens.bool().squeeze(), :])
#     disct2 = torch.mean(pred_label[~label_sens.bool().squeeze(), :])
#     print('discrimination', torch.abs(disct1-disct2))

