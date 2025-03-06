
import numpy as np
import torch
import torch.utils.data as utils
from sklearn import preprocessing
import pandas as pd
from scipy.io import loadmat
from collections import Counter

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def infer_dataloader(dataset_config):

    label_df = pd.read_csv(dataset_config["label"])


    if dataset_config["dataset"] == "PNC":
        fc_data = np.load(dataset_config["time_seires"], allow_pickle=True).item()
        fc_timeseires = fc_data['data'].transpose((0, 2, 1))

        fc_id = fc_data['id']
    

        id2gender = dict(zip(label_df['SUBJID'], label_df['sex']))

        final_fc, final_label = [], []

        for fc, l in zip(fc_timeseires, fc_id):
            if l in id2gender:
                final_fc.append(fc)
                final_label.append(id2gender[l])
        final_fc = np.array(final_fc)


    elif dataset_config["dataset"] == 'ABCD':

        fc_data = np.load(dataset_config["time_seires"], allow_pickle=True)

    _, node_size, timeseries = final_fc.shape

    encoder = preprocessing.LabelEncoder()

    encoder.fit(label_df["sex"])

    labels = encoder.transform(final_label)

    final_fc = torch.from_numpy(final_fc).float()

    return final_fc, labels, node_size, timeseries


        
def init_dataloader(dataset_config):

    if dataset_config["dataset"] == 'ABIDE':
        # np.load(保存文件名, allow_pickle=True)，这会返回一个object的ndarry格式的数据，
        # 后面再加一个.item()就可以变为字典格式
        # data={dict:5} 'timeseires'={ndarray:(1009,200,100)} 'label'={ndarray:(1009,)} 'corr'={ndarray:(1009,200,200)} 'pcorr'={ndarray:(1009,200,200)} ‘site’={ndarray:(1009,)}
        data = np.load(dataset_config["time_seires"], allow_pickle=True).item()
        final_fc = data["timeseires"]
        final_pearson = data["corr"]
        labels = data["label"]
        subject_id = data["id"]

    elif dataset_config["dataset"] == 'RD-TD':
        # np.load(保存文件名, allow_pickle=True)，这会返回一个object的ndarry格式的数据，
        # 后面再加一个.item()就可以变为字典格式
        # data={dict:5} 'timeseires'={ndarray:(42,116,100)} 'label'={ndarray:(42,)} 'corr'={ndarray:(42,116,116)} 'pcorr'={ndarray:(42,116,116)}}
        data = np.load(dataset_config["time_seires"], allow_pickle=True).item()
        final_fc = data["timeseires"]
        final_pearson = data["corr"]
        labels = data["label"]
        subject_id = data["id"]

    elif dataset_config["dataset"] == "HIV" or dataset_config["dataset"] == "BP":
        data = loadmat(dataset_config["node_feature"])

        labels = data['label']
        labels = labels.reshape(labels.shape[0])

        labels[labels==-1] = 0

        view = dataset_config["view"]

        final_pearson = data[view]

        final_pearson = np.array(final_pearson).transpose(2, 0, 1)

        final_fc = np.ones((final_pearson.shape[0],1,1))

    elif dataset_config["dataset"] == 'PPMI' or dataset_config["dataset"] == 'PPMI_balanced':
        m = loadmat(dataset_config["node_feature"])
        labels = m['label'] if dataset_config["dataset"] != 'PPMI_balanced' else m['label_new']
        labels = labels.reshape(labels.shape[0])
        data = m['X'] if dataset_config["dataset"] == 'PPMI' else m['X_new']
        final_pearson = np.zeros((data.shape[0], 84, 84))
        modal_index = 0
        for (index, sample) in enumerate(data):
            # Assign the first view in the three views of PPMI to a1
            final_pearson[index, :, :] = sample[0][:, :, modal_index]

        final_fc = np.ones((final_pearson.shape[0],1,1))

    else:

        fc_data = np.load(dataset_config["time_seires"], allow_pickle=True)
        pearson_data = np.load(dataset_config["node_feature"], allow_pickle=True)
        label_df = pd.read_csv(dataset_config["label"])

        if dataset_config["dataset"] == 'ABCD':

            with open(dataset_config["node_id"], 'r') as f:
                lines = f.readlines()
                pearson_id = [line[:-1] for line in lines]

            with open(dataset_config["seires_id"], 'r') as f:
                lines = f.readlines()
                fc_id = [line[:-1] for line in lines]

            id2pearson = dict(zip(pearson_id, pearson_data))

            id2gender = dict(zip(label_df['id'], label_df['sex']))

            final_fc, final_label, final_pearson = [], [], []

            for fc, l in zip(fc_data, fc_id):
                if l in id2gender and l in id2pearson:
                    if np.any(np.isnan(id2pearson[l])) == False:
                        final_fc.append(fc)
                        final_label.append(id2gender[l])
                        final_pearson.append(id2pearson[l])

            final_pearson = np.array(final_pearson)

            final_fc = np.array(final_fc)

        elif dataset_config["dataset"] == "PNC":
            pearson_data, fc_data = pearson_data.item(), fc_data.item()

            pearson_id = pearson_data['id']
            pearson_data = pearson_data['data']
            id2pearson = dict(zip(pearson_id, pearson_data))

            fc_id = fc_data['id']
            fc_data = fc_data['data']

            id2gender = dict(zip(label_df['SUBJID'], label_df['sex']))

            final_fc, final_label, final_pearson = [], [], []

            for fc, l in zip(fc_data, fc_id):
                if l in id2gender and l in id2pearson:
                    final_fc.append(fc)
                    final_label.append(id2gender[l])
                    final_pearson.append(id2pearson[l])

            final_pearson = np.array(final_pearson)

            final_fc = np.array(final_fc).transpose(0, 2, 1)

    _, _, timeseries = final_fc.shape # 100

    _, node_size, node_feature_size = final_pearson.shape
    print("node_size:",node_size) # 116
    print("node_feature_size",node_feature_size) # 116

    scaler = StandardScaler(mean=np.mean(
        final_fc), std=np.std(final_fc))
    
    final_fc = scaler.transform(final_fc) # (data - self.mean) / self.std

    if dataset_config["dataset"] == 'PNC' or dataset_config["dataset"] == 'ABCD':

        encoder = preprocessing.LabelEncoder()

        encoder.fit(label_df["sex"])

        labels = encoder.transform(final_label)

    final_fc, final_pearson, labels, subject_id = [torch.from_numpy(
        data).float() for data in (final_fc, final_pearson, labels, subject_id)] # 数据格式由ndarray转为tensor

    length = final_fc.shape[0] # 样本数量42
    print("样本数量: ",length)
    # train_length = int(length*dataset_config["train_set"])
    # print("train_length: ",train_length) # 29
    # val_length = int(length*dataset_config["val_set"])
    # print("val_length: ",val_length) # 0

    dataset = utils.TensorDataset(
        final_fc,
        final_pearson,
        labels,
        subject_id
    )

    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, [train_length, val_length, length-train_length-val_length])
    #
    # train_dataloader = utils.DataLoader(
    #     train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)
    #
    # val_dataloader = utils.DataLoader(
    #     val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)
    #
    # test_dataloader = utils.DataLoader(
    #     test_dataset, batch_size=dataset_config["test_batch_size"], shuffle=True, drop_last=False)


    # 按标签分割数据子集
    label_counter = Counter(labels.numpy())
    label_0_indices = [i for i, label in enumerate(labels) if label == 0]
    label_1_indices = [i for i, label in enumerate(labels) if label == 1]

    # 确定性划分子集
    label_0_length = label_counter[0]
    label_1_length = label_counter[1]

    train_length_0 = int(label_0_length * dataset_config["train_set"])
    val_length_0 = int(label_0_length * dataset_config["val_set"])

    train_length_1 = int(label_1_length * dataset_config["train_set"])
    val_length_1 = int(label_1_length * dataset_config["val_set"])


    print("train_length_0: ", train_length_0)
    print("val_length_0: ", val_length_0)
    print("train_length_0: ", train_length_1)
    print("val_length_0: ", val_length_1)

    # 创建最终的train_dataset、val_dataset和test_dataset
    train_indices = label_0_indices[:train_length_0] + label_1_indices[:train_length_1]
    val_indices = label_0_indices[train_length_0:train_length_0 + val_length_0] + label_1_indices[
                                                                                  train_length_1:train_length_1 + val_length_1]
    test_indices = label_0_indices[train_length_0 + val_length_0:] + label_1_indices[train_length_1 + val_length_1:]
    print("train_indices长度:", len(train_indices))
    print("val_indices长度:", len(val_indices))
    print("test_indices长度:", len(test_indices))

    train_dataset = utils.Subset(dataset, train_indices)
    val_dataset = utils.Subset(dataset, val_indices)
    test_dataset = utils.Subset(dataset, test_indices)

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=dataset_config["test_batch_size"], shuffle=True, drop_last=False)

    return (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries
