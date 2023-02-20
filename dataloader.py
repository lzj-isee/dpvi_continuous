import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from libsvm.python.svmutil import svm_read_problem
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import scipy.io as scio

class myDataLoader(object):
    def __init__(self, opts) -> None:
        self.opts = opts
        self.device = opts.device
        # load dataset
        if opts.task == 'logistic_regression': 
            self.load_dataset_lr(opts.dataset)
        elif opts.task == 'bnn_regression':
            self.n_hidden = 50
            self.load_dataset_nn(opts.dataset)
        elif opts.task == 'bnn_classification': 
            self.load_dataset_cl(opts.dataset)
        elif opts.task == 'gaussian_process':
            self.load_dataset_gp(opts.dataset)
        elif opts.task == 'demo':
            pass
        elif opts.task == 'funnel':
            pass
        elif opts.task == 'single_gaussian':
            pass
        elif opts.task == 'multi_gaussian':
            pass
        elif opts.task == 'ica_meg':
            self.load_dataset_meg(opts.dataset)
        else:
            raise NotImplementedError
    
    def get_train_loader(self):
        # set dataloader, 
        if self.opts.task == 'logistic_regression': # fake dataloader, keep code consistent
            self.train_loader = [[0, 0]]
        elif self.opts.task in ['ica', 'ica_meg']:
            self.train_loader = [[0, 0]]
        elif self.opts.task == 'gaussian_process':
            self.train_loader = [[0, 0]]
        elif self.opts.task in ['bnn_regression', 'bnn_classification']:
            train_set = TensorDataset(self.train_features, self.train_labels)
            self.train_loader = DataLoader(
                dataset = train_set,
                batch_size = self.opts.batch_size,
                shuffle = True,
                drop_last = True
            )
        elif self.opts.task in ['funnel', 'single_gaussian', 'multi_gaussian', 'demo']:
            self.train_loader = [[0, 0]]
        else:
            raise NotImplementedError
        return self.train_loader

    def load_dataset_gp(self, dataset_name):
        data_file = os.path.join('datasets', dataset_name, dataset_name + '.mat')
        raw_data = scio.loadmat(data_file)
        self.train_features = torch.from_numpy(raw_data['range'].astype(np.float32)).to(self.device) # N * 1 array
        self.train_labels = torch.from_numpy(raw_data['logratio'].astype(np.float32)).to(self.device) # N * 1 array
        self.test_features = None
        self.test_labels = None
        self.model_dim = 2
        self.train_num = self.train_features.shape[0]

    def load_dataset_meg(self, dataset_name):
        lines = []
        with open(os.path.join('datasets', dataset_name, dataset_name + '.txt')) as f:
            for i in range(self.opts.dim):
                lines.append(list(map(float,f.readline().strip().split('  '))))
        data_raw = np.array(lines)
        train_features, test_features = train_test_split(data_raw.transpose(), train_size = self.opts.train_num, random_state = self.opts.split_seed)
        self.train_features = torch.from_numpy(train_features).float().to(self.device)
        self.test_features = torch.from_numpy(test_features).float().to(self.device)
        self.model_dim = self.opts.dim ** 2
        self.train_num = self.opts.train_num
        self.test_num = self.test_features.shape[0]
        self.train_labels = torch.zeros(self.train_num, device = self.device) # fake labels
        #self.test_features /= 1000
        #self.train_features /= 1000
        std_features = torch.std(self.train_features, dim = 0, unbiased = True)
        std_features[std_features == 0] = 1
        mean_features = torch.mean(self.train_features, dim = 0)
        self.train_features = (self.train_features - mean_features) / std_features
        self.test_features = (self.test_features - mean_features) / std_features

    def load_dataset_cl(self, dataset_name):
        main_folder = os.path.join('datasets')
        if dataset_name in 'usps': 
            train_path = os.path.join(main_folder, dataset_name, dataset_name + '-train.txt')
            test_path = os.path.join(main_folder, dataset_name, dataset_name + '-test.txt')
            train_labels_raw, train_features_raw = svm_read_problem(train_path, return_scipy = True)
            test_labels_raw, test_features_raw = svm_read_problem(test_path, return_scipy = True)
            if dataset_name in 'usps': 
                train_labels_raw = train_labels_raw - 1
                test_labels_raw = test_labels_raw - 1
            self.num_classes = int(train_labels_raw.max() + 1)
        else: 
            raise ValueError('dataset {} not found'.format(dataset_name))
        train_features_raw = train_features_raw.toarray()
        test_features_raw = test_features_raw.toarray()
        # to Tensor
        self.train_features = torch.from_numpy(train_features_raw).float().to(self.device)
        self.train_labels =  torch.nn.functional.one_hot(torch.from_numpy(train_labels_raw).long(), self.num_classes).to(self.device)
        self.test_features = torch.from_numpy(test_features_raw).float().to(self.device)
        self.test_labels =  torch.nn.functional.one_hot(torch.from_numpy(test_labels_raw).long(), self.num_classes).to(self.device)
        # record information
        self.data_dim = len(self.train_features[0])
        self.train_num = len(self.train_features)
        self.test_num = len(self.test_features)
        self.model_dim = self.data_dim * self.n_hidden + self.n_hidden * self.num_classes +\
            self.n_hidden + self.num_classes
        

    def load_dataset_lr(self, dataset_name):
        main_folder = os.path.join('datasets')
        # load and split dataset
        if dataset_name in 'a3a, a9a, ijcnn, gisette, w8a, a8a, codrna, madelon':  # no need to split dataset
            train_path = os.path.join(main_folder, dataset_name, dataset_name + '-train.txt')
            test_path = os.path.join(main_folder, dataset_name, dataset_name + '-test.txt')
            train_labels_raw, train_features_raw = svm_read_problem(train_path, return_scipy = True)
            test_labels_raw, test_features_raw = svm_read_problem(test_path, return_scipy = True)
        elif dataset_name in 'mushrooms, pima, covtype, phishing, susy, fourclass, heart':    # split dataset
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.txt')
            labels_raw, features_raw = svm_read_problem(data_path, return_scipy = True)
            if dataset_name in 'mushrooms, covtype': labels_raw = (labels_raw - 1.5) * 2
            if dataset_name in 'phishing, susy': labels_raw = (labels_raw - 0.5) * 2
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.split_size, random_state = self.opts.split_seed)
        else:
            raise ValueError('dataset {} not found'.format(dataset_name))
        # some extra process for certain dataset
        train_features_raw = train_features_raw.toarray()
        test_features_raw = test_features_raw.toarray()
        self.train_num = len(train_labels_raw)
        self.test_num = len(test_labels_raw)
        if dataset_name == 'a3a':
            train_features_raw = np.concatenate((train_features_raw, np.zeros([self.train_num,1])), axis=1)
        if dataset_name in 'a9a, a8a':
            test_features_raw = np.concatenate((test_features_raw, np.zeros([self.test_num,1])), axis=1)
        # to Tensor
        self.train_features = torch.from_numpy(train_features_raw).float().to(self.device)
        self.train_labels = torch.from_numpy(train_labels_raw).float().to(self.device)
        self.test_features = torch.from_numpy(test_features_raw).float().to(self.device)
        self.test_labels = torch.from_numpy(test_labels_raw).float().to(self.device)
        self.data_dim = len(self.train_features[0])
        self.model_dim = self.data_dim + 1 # + 1 for bias
        # scale
        std_features = torch.std(self.train_features, dim = 0, unbiased = True)
        std_features[std_features == 0] = 1
        mean_features = torch.mean(self.train_features, dim = 0)
        self.train_features = (self.train_features - mean_features) / std_features
        self.test_features = (self.test_features - mean_features) / std_features
        # concatenate bias
        self.train_features = torch.cat([torch.ones(self.train_num, 1, device = self.device), self.train_features], dim = 1)
        self.test_features = torch.cat([torch.ones(self.test_num, 1, device = self.device), self.test_features], dim = 1)

    def load_dataset_nn(self, dataset_name):
        main_folder = os.path.join('datasets')
        self.out_dim = 1
        # load and split dataset
        if dataset_name in 'YearPredictionMSD':  # no need to split dataset
            train_path = os.path.join(main_folder, dataset_name, dataset_name + '-train.txt')
            test_path = os.path.join(main_folder, dataset_name, dataset_name + '-test.txt')
            train_labels_raw, train_features_raw = svm_read_problem(train_path, return_scipy = True)
            test_labels_raw, test_features_raw = svm_read_problem(test_path, return_scipy = True)
            train_features_raw = train_features_raw.toarray()
            test_features_raw = test_features_raw.toarray()
        elif dataset_name in 'abalone, boston, mpg, cpusmall, cadata, space, mg':    # split dataset
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.txt')
            labels_raw, features_raw = svm_read_problem(data_path, return_scipy = True)
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.split_size, random_state = self.opts.split_seed)
            train_features_raw = train_features_raw.toarray()
            test_features_raw = test_features_raw.toarray()
        elif dataset_name in 'concrete':    # load xls file and split the dataset
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.xls')
            data_raw = pd.read_excel(data_path, header = 0).values
            labels_raw, features_raw = data_raw[:, -1], data_raw[:,:-1]
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.split_size, random_state = self.opts.split_seed)
        elif dataset_name in 'energy, kin8nm, casp, superconduct, slice, online, sgemm, electrical, churn':  # load csv file and split the dataset
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.csv')
            data_raw = pd.read_csv(data_path, header = 0).values
            if dataset_name in 'energy':
                labels_raw, features_raw = data_raw[:, 1].astype(np.float32), data_raw[:, 2:].astype(np.float32)
            elif dataset_name in 'kin8nm, superconduct, slice':
                labels_raw, features_raw = data_raw[:, -1], data_raw[:, :-1]
            elif dataset_name in 'casp':
                labels_raw, features_raw = data_raw[:, 0], data_raw[:, 1:]
            elif dataset_name in 'online':
                labels_raw, features_raw = data_raw[:, -1].astype(np.float32), data_raw[:, 1:-1].astype(np.float32)
            elif dataset_name in 'sgemm':
                labels_raw, features_raw = data_raw[:, -4], data_raw[:, :-4]
            elif dataset_name in 'electrical':
                labels_raw, features_raw = data_raw[:, -2].astype(np.float32), data_raw[:, :-2].astype(np.float32)
            elif dataset_name in 'churn':
                labels_raw, features_raw = data_raw[:, -1].astype(np.float32), data_raw[:, 1:-1].astype(np.float32)
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.split_size, random_state = self.opts.split_seed)    
        elif dataset_name in 'WineRed, WineWhite':
            data_path = os.path.join(main_folder, dataset_name, dataset_name + '.csv')
            attris = pd.read_csv(data_path, header = 0).values.reshape(-1).tolist()
            data_raw = []
            for attr in attris:
                data_raw.append([eval(number) for number in attr.split(';')])
            data_raw = np.array(data_raw)
            labels_raw, features_raw = data_raw[:, -1], data_raw[:,:-1]
            train_features_raw, test_features_raw, train_labels_raw, test_labels_raw = train_test_split(
                features_raw, labels_raw, test_size = self.opts.split_size, random_state = self.opts.split_seed)  
        else:   # TODO:  naval
            raise ValueError('dataset {} not found'.format(dataset_name))
        # to Tensor
        self.train_features = torch.from_numpy(train_features_raw).float().to(self.device)
        self.train_labels = torch.from_numpy(train_labels_raw).float().to(self.device)
        self.test_features = torch.from_numpy(test_features_raw).float().to(self.device)
        self.test_labels = torch.from_numpy(test_labels_raw).float().to(self.device)
        # Normalization
        self.std_features = torch.std(self.train_features, dim = 0, unbiased = True)
        self.std_features[self.std_features == 0] = 1
        self.mean_features = torch.mean(self.train_features, dim = 0)
        self.std_labels = torch.std(self.train_labels, dim = 0)
        self.mean_labels = torch.mean(self.train_labels, dim = 0)
        self.train_features = (self.train_features - self.mean_features) / self.std_features
        self.train_labels = (self.train_labels - self.mean_labels) / self.std_labels
        self.test_features = (self.test_features - self.mean_features) / self.std_features
        # self.test_labels = (self.test_labels - self.mean_labels) / self.std_labels
        # record information
        self.data_dim = len(self.train_features[0])
        self.train_num = len(self.train_features)
        self.test_num = len(self.test_features)
        self.model_dim = self.data_dim * self.n_hidden + self.n_hidden * self.out_dim +\
            self.n_hidden + self.out_dim + 2  # 2 variances
