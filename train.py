import numpy as np
import argparse
import torch
from pylab import rcParams
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os.path
import sys
import math
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torch.utils.data as data_utils
from torch import autograd
import time
import torchvision
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from model import Model, reparam, KL, negative_log_bernoulli, entropy_gaussian, negative_log_gaussian, kernel
from sklearn.manifold import TSNE
import torch.nn as nn
import datetime
import argparse
import os
from preprocess_adult import load_adult

def imshow(img):
    plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))


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


def main():
    parser = argparse.ArgumentParser(description='VAE+VampPrior')
    # arguments for optimization
    parser.add_argument('--train_epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--latent_size', default=50, type=int)
    parser.add_argument('--input_size', default=784, type=int)
    parser.add_argument('--test_epochs', default=200, type=int)
    parser.add_argument('--model_name', default='VFIB', type=str)
    parser.add_argument('--with_mmd', default=False, action='store_true')
    parser.add_argument('--train_num', default=1, type=int)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--sensitive_attr', default=-1, type=int)

    config = parser.parse_args()

    if config.model_name == 'VFIB':
        beta = 1
    elif config.model_name == 'VFAE':
        beta = 1
    elif config.model_name == 'LCFR':
        beta = 1
    elif config.model_name == 'VAE':
        beta = 1
    elif config.model_name == 'VFIBG':
        beta = 50
        beta2 = 50

    if config.dataset == 'mnist':
        train_data, train_label, test_data, test_label = load_mnist()
    elif config.dataset == 'adult':
        train_data, train_label, test_data, test_label = load_adult()
        if config.model_name is not "VFIB":
            threshold = [38.64, 189664.13, 10.07, 1079.06, 87.502314, 40.422382]
            for i in range(len(threshold)):
                train_data[:, i] = (train_data[:, i] > threshold[i]).double()
                test_data[:, i] = (test_data[:, i] > threshold[i]).double()
            print(train_data[:10])

    config.input_size = 109 if config.dataset == 'adult' else 784
    config.sensitive_attr = 0 if config.dataset == 'adult' else -1

    train_dataset = data_utils.TensorDataset(train_data, train_label)
    test_dataset = data_utils.TensorDataset(test_data, test_label)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=20000, shuffle=True)

    print(train_data.shape)
    print(test_data.shape)


    for train_n in range(config.train_num):

        model = Model(input_size=config.input_size, latent_size=config.latent_size, model_name=config.model_name)
        model.train()
        model.double()
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)


        for epoch_number in range(config.train_epochs):
            print('epoch number:', epoch_number)
            # with autograd.detect_anomaly():
            for (data, label) in train_loader:
                label = label.double().cuda()
                data = data.double().cuda()
                sens_attr = data[:, config.sensitive_attr].unsqueeze(1)

                # print(model.classifier[0].weight)
                # print(label)
                # imshow(torchvision.utils.make_grid(data[:, :-1].view(128, 1, 28, 28)))
                # plt.show()
                # print('sens attr', sens_attr.view(-1))
                # print('label', label.view(-1))
                # imshow(torchvision.utils.make_grid(data[:128, :-1].view(128, 1, 28, 28)))
                # plt.show()

                if config.model_name == 'VFIB': #Proposed
                    q_z_mean, q_z_log_sigma = model.encoder(data)
                    z = reparam(q_z_mean, q_z_log_sigma)
                    pred_mean = model.classifier(torch.cat((z, sens_attr), dim=1))
                    classifier_loss = negative_log_bernoulli(label, pred_mean)
                    kl_loss = KL(q_z_mean, q_z_log_sigma)

                    z_s_0 = z[sens_attr.bool().squeeze(), :]
                    z_s_1 = z[~sens_attr.bool().squeeze(), :]

                    mmd_loss = kernel(z_s_0, z_s_0) + kernel(z_s_1, z_s_1) - 2 * kernel(z_s_0, z_s_1)

                    loss = beta * kl_loss + classifier_loss

                    if config.with_mmd:
                        loss += mmd_loss

                if config.model_name == 'VFIBG':

                    q_z_mean, q_z_log_sigma = model.encoder(data)
                    z = reparam(q_z_mean, q_z_log_sigma)
                    pred_mean = model.classifier(torch.cat((z, sens_attr), dim=1))

                    classifier_loss = negative_log_bernoulli(label, pred_mean)

                    kl_loss = KL(q_z_mean, q_z_log_sigma)

                    reconst = model.decoder(torch.cat((z, sens_attr), dim=1))
                    reconst_loss = negative_log_bernoulli(data[:, :-1], reconst)

                    loss = beta * kl_loss + beta2*classifier_loss + reconst_loss

                elif config.model_name == 'LCFR': #Unsupervised version of VFAE
                    q_z_mean, q_z_log_sigma = model.encoder(data)
                    z = reparam(q_z_mean, q_z_log_sigma)
                    reconst = model.decoder(torch.cat((z, sens_attr), dim=1))
                    reconst_loss = negative_log_bernoulli(data[:, :-1], reconst)

                    kl_loss = KL(q_z_mean, q_z_log_sigma)
                    loss = beta*kl_loss + reconst_loss

                elif config.model_name == 'VAE': #Original VAE
                    q_z_mean, q_z_log_sigma = model.encoder(data)
                    z = reparam(q_z_mean, q_z_log_sigma)
                    reconst = model.decoder(z)
                    reconst_loss = negative_log_bernoulli(data[:, :-1], reconst)
                    # reconst_loss = torch.sum((data[:, :-1] - reconst)**2, dim=1).mean()
                    kl_loss = KL(q_z_mean, q_z_log_sigma)
                    loss = beta * kl_loss + reconst_loss

                elif config.model_name == 'VFAE':
                    q_z_mean, q_z_log_sigma = model.encoder(data)
                    z = reparam(q_z_mean, q_z_log_sigma)
                    q_z_1_mean, q_z_1_log_sigma = model.encoder_z(torch.cat((z, label), dim=1))
                    z_1 = reparam(q_z_1_mean, q_z_1_log_sigma)

                    kl_loss = KL(q_z_1_mean, q_z_1_log_sigma)

                    reconst = model.decoder(torch.cat((z, sens_attr), dim=1))
                    reconst_loss = negative_log_bernoulli(data[:, :-1], reconst)

                    z_reconst_mean, z_reconst_log_sigma = model.reconst_z(torch.cat((z_1, label), dim=1))
                    reconst_z_loss = negative_log_gaussian(z, z_reconst_mean, z_reconst_log_sigma)

                    pred_mean = model.classifier(z)
                    classifier_loss = negative_log_bernoulli(label, pred_mean)

                    entropy_z = entropy_gaussian(q_z_mean, q_z_log_sigma)

                    z_s_0 = z[sens_attr.bool().squeeze(), :]
                    z_s_1 = z[~sens_attr.bool().squeeze(), :]
                    mmd_loss = kernel(z_s_0, z_s_0) + kernel(z_s_1, z_s_1) - 2 * kernel(z_s_0, z_s_1)

                    loss = reconst_loss + kl_loss + reconst_z_loss - entropy_z + beta*classifier_loss

                    if config.with_mmd:
                        loss += mmd_loss

                model.zero_grad()
                loss.backward()
                optimizer.step()
                # beta += 0.1
            print('batch loss:', loss)
            if config.model_name is not 'VAE' and config.model_name is not 'LCFR':
                counts = ((pred_mean.detach() > 0.5).double() == label).double()
                print('batch accuracy:', torch.mean(counts))

        model.eval()
        sigmoid = torch.nn.Sigmoid()
        test_classifier = nn.Sequential(nn.Linear(config.latent_size, 1)).double().cuda()
        sens_classifier = nn.Sequential(nn.Linear(config.latent_size, 1)).double().cuda()

        test_classifier_deter = nn.Sequential(nn.Linear(config.latent_size, 1)).double().cuda()
        sens_classifier_deter = nn.Sequential(nn.Linear(config.latent_size, 1)).double().cuda()

        test_decoder = nn.Sequential(nn.Linear(config.latent_size, 100), nn.Tanh(), nn.Linear(100, config.input_size)).double().cuda()
        test_decoder_sens = nn.Sequential(nn.Linear(config.latent_size+1, 100), nn.Tanh(), nn.Linear(100, config.input_size)).double().cuda()

        optimizer_label = torch.optim.Adam(test_classifier.parameters(), lr=config.lr)
        optimizer_sens = torch.optim.Adam(sens_classifier.parameters(), lr=config.lr)
        optimizer_decoder = torch.optim.Adam(test_decoder.parameters(), lr=config.lr)
        optimizer_decoder_sens = torch.optim.Adam(test_decoder_sens.parameters(), lr=config.lr)
        optimizer_label_deter = torch.optim.Adam(test_classifier_deter.parameters(), lr=config.lr)
        optimizer_sens_deter = torch.optim.Adam(sens_classifier_deter.parameters(), lr=config.lr)

        for epoch_number in range(config.test_epochs):
            print('epoch number:', epoch_number)
            for (data, label) in train_loader:
                label = label.double().cuda()
                data = data.double().cuda()
                label_sens = data[:, config.sensitive_attr].unsqueeze(1)
                # print('label sens shape', label_sens.shape)

                q_z_mean, q_z_log_variance = model.encoder(data)
                q_z_mean = q_z_mean.detach()
                q_z_log_variance = q_z_log_variance.detach()
                z = reparam(q_z_mean, q_z_log_variance)

                pred_label_deter = test_classifier_deter(q_z_mean)
                pred_sens_deter = sens_classifier_deter(q_z_mean)

                pred_label = test_classifier(z)
                pred_sens = sens_classifier(z)

                reconst = test_decoder(z)
                reconst_sens = test_decoder_sens(torch.cat((z, label_sens), dim=1))
                # print(pred_label.shape)
                # print(pred_sens.shape)

                label_loss = negative_log_bernoulli(label, pred_label)
                sens_loss = negative_log_bernoulli(label_sens, pred_sens)

                label_loss_deter = negative_log_bernoulli(label, pred_label_deter)
                sens_loss_deter = negative_log_bernoulli(label_sens, pred_sens_deter)

                reconst_loss = negative_log_bernoulli(data[:, :-1], reconst)
                reconst_sens_loss = negative_log_bernoulli(data[:, :-1], reconst_sens)

                # optimizers
                optimizer_label.zero_grad()
                optimizer_sens.zero_grad()

                optimizer_label_deter.zero_grad()
                optimizer_sens_deter.zero_grad()

                optimizer_decoder.zero_grad()
                optimizer_decoder_sens.zero_grad()

                # backward
                label_loss.backward()
                sens_loss.backward()

                label_loss_deter.backward()
                sens_loss_deter.backward()

                reconst_loss.backward()
                reconst_sens_loss.backward()

                # optimizers step
                optimizer_label.step()
                optimizer_sens.step()

                optimizer_label_deter.step()
                optimizer_sens_deter.step()

                optimizer_decoder.step()
                optimizer_decoder_sens.step()

            acc_label = torch.mean(((sigmoid(pred_label).detach() > 0.5).double() == label).double())
            acc_sens = torch.mean(((sigmoid(pred_sens).detach() > 0.5).double() == label_sens).double())

            print('last train batch acc_label:', acc_label)
            print('last train batch acc_sens:', acc_sens)

        dir = 'trained_models/' + config.model_name + '/' + config.dataset + '/' + str(datetime.datetime.now()).replace(" ", "-") +\
              "_" + config.model_name + '/'

        os.mkdirs(dir, exist_ok=True)

        for (data, label) in test_loader:
            data = data.double().cuda()
            label = label.double().cuda()

            label_sens = data[:, config.sensitive_attr].unsqueeze(1)

            q_z_mean, q_z_log_sigma = model.encoder(data)
            q_z_mean = q_z_mean.detach()
            q_z_log_variance = q_z_log_variance.detach()
            z = reparam(q_z_mean, q_z_log_sigma)

            # Prediction
            pred_label = test_classifier(z)
            pred_sens = sens_classifier(z)

            pred_label_deter = test_classifier_deter(q_z_mean)
            pred_sens_deter = sens_classifier_deter(q_z_mean)

            acc_label_deter = torch.mean(((sigmoid(pred_label_deter).detach() > 0.5).double() == label).double())
            acc_sens_deter = torch.mean(((sigmoid(pred_sens_deter).detach() > 0.5).double() == label_sens).double())

            acc_label = torch.mean(((sigmoid(pred_label).detach() > 0.5).double() == label).double())
            acc_sens = torch.mean(((sigmoid(pred_sens).detach() > 0.5).double() == label_sens).double())

            print('test acc label:', acc_label)
            print('test acc sens:', acc_sens)

            print('test acc label deter:', acc_label_deter)
            print('test acc sens deter:', acc_sens_deter)

            with open(dir+"evaluation.txt", "w") as file:
                acc_label_txt = 'test acc label: ' + str(acc_label.item())
                acc_sens_txt = 'test acc sens: ' + str(acc_sens.item())
                acc_label_txt_deter = 'test acc label deter: ' + str(acc_label_deter.item())
                acc_sens_txt_deter = 'test acc sens deter: ' + str(acc_sens_deter.item())
                file.write(acc_label_txt +
                           '\n' + acc_sens_txt +
                           '\n' + acc_label_txt_deter +
                           '\n' + acc_sens_txt_deter)

            with open(dir+"config.txt", "w") as file:
                file.write(str(config))

            rcParams['figure.figsize'] = 10, 10
            if datasets == 'mnist':
                reconst = sigmoid(test_decoder(z[:64]).detach())
                reconst_sens = sigmoid(test_decoder_sens(torch.cat((z[:64], label_sens[:64]), dim=1)).detach())

                plt.subplot(311)
                imshow(torchvision.utils.make_grid(data[:64, :-1].view(64, 1, 28, 28)))

                plt.subplot(312)
                imshow(torchvision.utils.make_grid(reconst.view(64, 1, 28, 28)))

                plt.subplot(313)
                imshow(torchvision.utils.make_grid(reconst_sens.view(64, 1, 28, 28)))
                plt.savefig(dir+'reconst')

            embeded_data = TSNE(n_components=2).fit_transform(q_z_mean[:2000].detach().cpu().numpy())
            colors_label = np.array(['orange', 'green'])
            colors_sens = np.array(['red', 'blue'])

            label = label[:2000].view(-1).detach().cpu().numpy().astype('int')
            label_legend = ['label=1', 'label=0']
            sens_attr = data[:2000, config.sensitive_attr].view(-1).detach().cpu().numpy().astype('int')
            sens_legend = ['sens=1', 'sens=0']

            plt.subplot(211)
            plt.scatter(embeded_data[:, 0], embeded_data[:, 1], s=1, c=colors_label[label], label=label_legend)
            plt.subplot(212)
            plt.scatter(embeded_data[:, 0], embeded_data[:, 1], s=1, c=colors_sens[sens_attr], label=sens_legend)
            plt.savefig(dir+'latent')

            torch.save(model, dir+'model')


if __name__ == '__main__':
    main()
