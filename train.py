from typing import overload

import matplotlib.pyplot as plt
import torch
from numpy.lib import save
from util import Logger, accuracy, TotalMeter
import numpy as np
import os
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from util.prepossess import mixup_criterion, mixup_data
from util.loss import mixup_cluster_loss
import math
import csv
from sklearn.manifold import TSNE
import warnings
import matplotlib.pyplot as plt

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

import numpy as np


def add_salt_and_pepper_noise(data, noise_ratio):
    noisy_data = np.copy(data)
    num_pixels = data.size
    num_noisy_pixels = int(noise_ratio * num_pixels)

    # 随机选择要添加噪声的像素位置
    noisy_pixels = np.random.choice(num_pixels, num_noisy_pixels, replace=False)

    # 将选择的像素位置设为最大或最小值，以添加椒盐噪声
    noisy_data.flat[noisy_pixels] = np.random.choice([0, 1], num_noisy_pixels)

    return noisy_data

class BasicTrain:

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        self.logger = Logger()
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = train_config['epochs']
        self.optimizers = optimizers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.sim_loss = train_config['sim_loss']
        self.group_loss = train_config['group_loss']

        self.sparsity_loss = train_config['sparsity_loss']
        self.sparsity_loss_weight = train_config['sparsity_loss_weight']
        self.sim_loss_weight = train_config['sim_loss_weight']
        self.save_path = log_folder

        self.save_learnable_graph = True

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy, \
        self.val_accuracy, self.test_accuracy, self.edges_num = [
            TotalMeter() for _ in range(7)]

        self.loss1, self.loss2, self.loss3 = [TotalMeter() for _ in range(3)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy, self.test_accuracy,
                      self.train_loss, self.val_loss, self.test_loss, self.edges_num,
                      self.loss1, self.loss2, self.loss3]:
            meter.reset()

    def mean2(self, x):
        y = np.sum(x) / np.size(x)
        return y

    def corr2(self, a, b):
        a = a - self.mean2(a)
        b = b - self.mean2(b)

        r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
        return r

    def matrixcorr2(self, matrix):
        matrix = matrix.cpu().detach().numpy()
        bz = matrix.shape[0]
        x = np.zeros((bz, bz))
        for i in range(bz):
            A = matrix[i, :, :]
            for j in range(i, bz):
                if (i == j):
                    x[i][j] = 1
                else:
                    B = matrix[j, :, :]
                    x[i][j] = self.corr2(A, B)
        x = np.triu(x)
        x += x.T - np.diag(x.diagonal())

        return x

    def peoplecorr2(self, subject_list, file_path):
        people_list = []
        np.seterr(divide='ignore', invalid='ignore')

        with open(file_path, 'r', encoding='UTF-8') as load_input:  # 打开要读取的csv文件进行只读操作
            ereader = csv.reader(load_input)  # 用reader函数读入文件指针
            eheader = next(ereader)  # 取出文件的第一行，也就是表头
            # 1表示男性，0表示女性
            for subject_id in subject_list:
                for row_list in ereader:
                    if int(str(row_list[0]).strip()) == subject_id:
                        people_list.append(row_list[1:])
                        break
                load_input.seek(0)  # 重置文件指针到文件开头
                next(ereader)  # 跳过标题行

        data = np.array(people_list)
        data = data.astype(float)
        pMatric = np.corrcoef(data)
        pMatric = np.nan_to_num(pMatric, copy=False)

        return pMatric

    def train_per_epoch(self, optimizer):

        self.model.train()
        for data_in, pearson, label, subject_id in self.train_dataloader:
            label = label.long()

            data_in, pearson, label, subject_id = data_in.to(device), pearson.to(device), label.to(
                device), subject_id.to(device)  # data_in=[16,200,100] pearson=[16,200,200] label=[16,]

            # mixup是一种数据增强技术，它可以通过将多组不同数据集的样本进行线性组合，生成新的样本，从而扩充数据集。
            # mixup的核心原理是将两个不同的图片按照一定的比例进行线性组合，生成新的样本，
            # 新样本的标签也是进行线性组合得到。
            inputs, nodes, targets_a, targets_b, lam = mixup_data(
                data_in, pearson, label, 0, device)  # inputs=[16,200,100] nodes=[16,200,200] targets_a=targets_b=[16,]

            # mu, sigma = 0, 1  # 均值和标准差
            # noise = np.random.normal(mu, sigma, inputs.shape)  # 生成高斯噪声
            # inputs = inputs.cpu().detach().numpy()
            # nodes = nodes.cpu().detach().numpy()
            # inputs = inputs + noise  # 加噪声
            # noise = np.random.normal(mu, sigma, nodes.shape)  # 生成高斯噪声
            # nodes = nodes + noise
            # inputs = torch.from_numpy(inputs).float().to(device)
            # nodes = torch.from_numpy(nodes).float().to(device)

            # noise_ratio = 0.1  # 噪声比例，即噪声像素占总像素的比例
            # inputs = inputs.cpu().detach().numpy()
            # nodes = nodes.cpu().detach().numpy()
            # inputs = add_salt_and_pepper_noise(inputs, noise_ratio)
            # nodes = add_salt_and_pepper_noise(nodes, noise_ratio)
            # inputs = torch.from_numpy(inputs).float().to(device)
            # nodes = torch.from_numpy(nodes).float().to(device)

            output, learnable_matrix, edge_variance,_ = self.model(inputs, nodes)

            # 增加 corr_x
            corr_x = self.matrixcorr2(learnable_matrix)
            # 增加 corr_y
            file_path = 'E:\\ADHD\\ADHD\\adhd200_preprocessed_phenotypics_normalized.csv'
                # 'E:\\ADHD\\ADHD\\adhd200_preprocessed_phenotypics_normalized.csv'
                # 'E:\ABIDE\ABIDE_pcp\Phenotypic_V1_0b_preprocessed1_normalized.csv'
                # '/mnt/5468e/ypzhang/LiYuxin/ABIDE/Phenotypic_V1_0b_preprocessed1_normalized.csv'
            subject_id2 = subject_id.tolist()
            corr_y = self.peoplecorr2(subject_id2, file_path)

            loss = 2 * mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            if self.sim_loss:
                loss += self.sim_loss_weight * (np.linalg.norm(corr_x - corr_y, 'fro') ** 2)

            if self.group_loss:
                loss += mixup_cluster_loss(learnable_matrix,
                                           targets_a, targets_b, lam)

            if self.sparsity_loss:
                sparsity_loss = self.sparsity_loss_weight * \
                                torch.norm(learnable_matrix, p=1)
                loss += sparsity_loss

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            self.edges_num.update_with_weight(edge_variance, label.shape[0])


    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for data_in, pearson, label, subject in dataloader:
            label = label.long()
            data_in, pearson, label, subject = data_in.to(
                device), pearson.to(device), label.to(device), subject.to(device)
            output, _, _, _ = self.model(data_in, pearson)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        fpr, tpr, thresholds = roc_curve(labels, result)  # 计算ROC曲线
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        # metric = precision_recall_fscore_support(
        #     labels, result,
        #     average='micro')  # precision_recall_fscore_support 直接把precision、recall、fscore和support的值都显示出来，其中support是每个标签在y_true中出现的次数
        acc = accuracy_score(labels, result) # 计算准确率Accuracy
        sen = recall_score(labels, result) # 计算灵敏度Sensitivity，即真阳性率True Positive Rate (TPR)
        ppv = precision_score(labels, result) # 计算精确率Positive Predictive Value (PPV)，即查准率
        f_score = f1_score(labels, result) # 计算F1分数
        mcc = matthews_corrcoef(labels, result) # 计算MCC
        # 特异性需要使用混淆矩阵计算
        tn, fp, fn, tp = confusion_matrix(labels, result).ravel()
        spe = recall_score(labels, result, pos_label=0) # 计算特异度Specificity，即真阴性率True Negative Rate (TNR)
        # return [auc] + list(metric)
        return [auc, fpr, tpr, thresholds, acc, sen, ppv, f_score, mcc, spe, tn, fp, fn, tp]

    def generate_save_learnable_matrix(self):
        learable_matrixs = []

        labels = []
        subjects = []
        data = []
        data2 = []
        for data_in, nodes, label, subject in self.train_dataloader:
            label = label.long()
            data_in, nodes, label, subject = data_in.to(
                device), nodes.to(device), label.to(device), subject.to(device)
            _, learable_matrix, _, embedding = self.model(data_in, nodes)
            data.append(embedding)
            data2.append(label)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()
            subjects += subject.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "learnable_train_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels), "subject": np.array(subjects)}, allow_pickle=True)

        return data, data2

    def save_result(self, results):
        self.save_path.mkdir(exist_ok=True, parents=True)
        # 转换数组类型的元素为 Numpy 数组
        result = [[np.array(item) if isinstance(item, list) else item for item in row] for row in results]

        # 获取结果序列的形状
        rows = len(result)
        cols = len(result[0])

        # 创建空的 Numpy 数组
        saved_array = np.empty((rows, cols), dtype=object)

        # 复制元素到数组中
        for i, row in enumerate(result):
            for j, item in enumerate(row):
                saved_array[i, j] = item

        np.save(self.save_path / "training_process.npy", saved_array, allow_pickle=True)

        # torch.save(self.model.state_dict(), self.save_path / "model.pt")

    def train(self):
        training_process = []
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Edges:{self.edges_num.avg: .3f}',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.2f}',
                f'Test AUC:{test_result[0]:.2f}'
            ]))
            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_accuracy.avg, self.test_loss.avg]
                                    + val_result + test_result)

        if self.save_learnable_graph:
            data , data2 = self.generate_save_learnable_matrix()
            A = data[0]
            C = data2[0]
            a = len(data)
            c = len(data2)
            for i in range(a):
                if i == len(data) - 1:
                    break
                else:
                    A = torch.cat([A, data[i + 1]], dim=0)
            for i in range(c):
                if i == c - 1:
                    break
                else:
                    C = torch.cat([C, data2[i + 1]], dim=0)
            A = A.detach().cpu().numpy()
            print(A.shape)
            C = C.detach().cpu().numpy()
            print(C.shape)
            self.save_path.mkdir(exist_ok=True, parents=True)
            np.save(self.save_path / "embedding.npy", A, allow_pickle=True)
            np.save(self.save_path / "label.npy", C, allow_pickle=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=100).fit_transform(A)

            # color1 = '#ee4433'
            # color2 = '#4488ff'
            # plt.scatter(tsne[:, 0], tsne[:, 1], c=np.where(C == 0, color1, color2), alpha=0.7)
            plt.scatter(tsne[:, 0], tsne[:, 1], c=C, alpha=0.7)
            # plt.scatter(tsne[:, 0], tsne[:, 1], c=C, alpha = 0.7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.savefig(os.path.join(self.save_path, "tsne.jpg"), bbox_inches="tight", dpi=800)
            # plt.show()
        self.save_result(training_process)


class BiLevelTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.save_learnable_graph = False

    def train(self):
        training_process = []
        matrix_epoch = 5

        for epoch in range(self.epochs):
            self.reset_meters()
            if epoch % 10 < matrix_epoch:
                self.train_per_epoch(self.optimizers[0])
            else:
                self.train_per_epoch(self.optimizers[1])
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Edges:{self.edges_num.avg: .3f}',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.2f}',
                f'Test AUC:{test_result[0]:.2f}'
            ]))
            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg]
                                    + val_result + test_result)
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)


class GNNTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.pure_gnn_graph = train_config['pure_gnn_graph']
        self.save_learnable_graph = False

    def train_per_epoch(self, optimizer):

        self.model.train()
        for _, pearson, label, _ in self.train_dataloader:
            label = label.long()

            pearson, label = pearson.to(device), label.to(device)

            bz, module_num, _ = pearson.shape

            if self.pure_gnn_graph == "uniform":
                graph = torch.ones(
                    (bz, module_num, module_num)).float().to(device)
            elif self.pure_gnn_graph == "pearson":
                graph = torch.abs(pearson)

            graph, nodes, targets_a, targets_b, lam = mixup_data(
                graph, pearson, label, 0, device)

            # mu, sigma = 0, 1  # 均值和标准差
            # # noise = np.random.normal(mu, sigma, graph.shape)  # 生成高斯噪声
            # # graph = graph.cpu().detach().numpy()
            # nodes = nodes.cpu().detach().numpy()
            # # graph = graph + noise  # 加噪声
            # noise = np.random.normal(mu, sigma, nodes.shape)  # 生成高斯噪声
            # nodes = nodes + noise
            # # graph = torch.from_numpy(graph).float().to(device)
            # nodes = torch.from_numpy(nodes).float().to(device)
            noise_ratio = 0.1  # 噪声比例，即噪声像素占总像素的比例
            graph = graph.cpu().detach().numpy()
            nodes = nodes.cpu().detach().numpy()
            graph = add_salt_and_pepper_noise(graph, noise_ratio)
            nodes = add_salt_and_pepper_noise(nodes, noise_ratio)
            graph = torch.from_numpy(graph).float().to(device)
            nodes = torch.from_numpy(nodes).float().to(device)

            output, _, _ = self.model(graph, nodes)

            loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for _, pearson, label, _ in dataloader:
            label = label.long()

            pearson, label = pearson.to(device), label.to(device)

            bz, module_num, _ = pearson.shape

            if self.pure_gnn_graph == "uniform":
                graph = torch.ones(
                    (bz, module_num, module_num)).float().to(device)
            elif self.pure_gnn_graph == "pearson":
                graph = torch.abs(pearson)

            output, _, _ = self.model(graph, pearson)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        # auc = roc_auc_score(labels, result)
        # result = np.array(result)
        # result[result > 0.5] = 1
        # result[result <= 0.5] = 0
        # metric = precision_recall_fscore_support(
        #     labels, result, average='micro')
        # return [auc] + list(metric)

        auc = roc_auc_score(labels, result)
        fpr, tpr, thresholds = roc_curve(labels, result)  # 计算ROC曲线
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        # metric = precision_recall_fscore_support(
        #     labels, result,
        #     average='micro')  # precision_recall_fscore_support 直接把precision、recall、fscore和support的值都显示出来，其中support是每个标签在y_true中出现的次数
        acc = accuracy_score(labels, result)  # 计算准确率Accuracy
        sen = recall_score(labels, result)  # 计算灵敏度Sensitivity，即真阳性率True Positive Rate (TPR)
        ppv = precision_score(labels, result)  # 计算精确率Positive Predictive Value (PPV)，即查准率
        f_score = f1_score(labels, result)  # 计算F1分数
        mcc = matthews_corrcoef(labels, result)  # 计算MCC
        # 特异性需要使用混淆矩阵计算
        tn, fp, fn, tp = confusion_matrix(labels, result).ravel()
        spe = recall_score(labels, result, pos_label=0)  # 计算特异度Specificity，即真阴性率True Negative Rate (TNR)
        # return [auc] + list(metric)
        return [auc, fpr, tpr, thresholds, acc, sen, ppv, f_score, mcc, spe, tn, fp, fn, tp]


class SeqTrain(BasicTrain):
    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.save_learnable_graph = False

    def train_per_epoch(self, optimizer):

        self.model.train()
        for seq_group, _, label, _ in self.train_dataloader:
            label = label.long()

            seq_group, label = seq_group.to(device), label.to(device)

            seq_group, _, targets_a, targets_b, lam = mixup_data(
                seq_group, seq_group, label, 0, device)

            # mu, sigma = 0, 1  # 均值和标准差
            # noise = np.random.normal(mu, sigma, seq_group.shape)  # 生成高斯噪声
            # seq_group = seq_group.cpu().detach().numpy()
            # seq_group = seq_group + noise  # 加噪
            # seq_group = torch.from_numpy(seq_group).float().to(device)
            noise_ratio = 0.1  # 噪声比例，即噪声像素占总像素的比例
            seq_group = seq_group.cpu().detach().numpy()
            seq_group = add_salt_and_pepper_noise(seq_group, noise_ratio)
            seq_group = torch.from_numpy(seq_group).float().to(device)

            output = self.model(seq_group)

            loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for seq_group, _, label, _ in dataloader:
            label = label.long()

            seq_group, label = seq_group.to(device), label.to(device)

            output = self.model(seq_group)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        # auc = roc_auc_score(labels, result)
        # result = np.array(result)
        # result[result > 0.5] = 1
        # result[result <= 0.5] = 0
        # metric = precision_recall_fscore_support(
        #     labels, result, average='micro')
        # return [auc] + list(metric)

        auc = roc_auc_score(labels, result)
        fpr, tpr, thresholds = roc_curve(labels, result)  # 计算ROC曲线
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        # metric = precision_recall_fscore_support(
        #     labels, result,
        #     average='micro')  # precision_recall_fscore_support 直接把precision、recall、fscore和support的值都显示出来，其中support是每个标签在y_true中出现的次数
        acc = accuracy_score(labels, result)  # 计算准确率Accuracy
        sen = recall_score(labels, result)  # 计算灵敏度Sensitivity，即真阳性率True Positive Rate (TPR)
        ppv = precision_score(labels, result)  # 计算精确率Positive Predictive Value (PPV)，即查准率
        f_score = f1_score(labels, result)  # 计算F1分数
        mcc = matthews_corrcoef(labels, result)  # 计算MCC
        # 特异性需要使用混淆矩阵计算
        tn, fp, fn, tp = confusion_matrix(labels, result).ravel()
        spe = recall_score(labels, result, pos_label=0)  # 计算特异度Specificity，即真阴性率True Negative Rate (TNR)
        # return [auc] + list(metric)
        return [auc, fpr, tpr, thresholds, acc, sen, ppv, f_score, mcc, spe, tn, fp, fn, tp]


class BrainCNNTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.save_learnable_graph = False

    def train_per_epoch(self, optimizer):

        self.model.train()

        for _, pearson, label,_ in self.train_dataloader:
            label = label.long()

            pearson, label = pearson.to(device), label.to(device)

            _, nodes, targets_a, targets_b, lam = mixup_data(
                pearson, pearson, label, 0, device)

            # mu, sigma = 0, 1  # 均值和标准差
            # nodes = nodes.cpu().detach().numpy()
            # noise = np.random.normal(mu, sigma, nodes.shape)  # 生成高斯噪声
            # nodes = nodes + noise
            # nodes = torch.from_numpy(nodes).float().to(device)
            noise_ratio = 0.1  # 噪声比例，即噪声像素占总像素的比例
            nodes = nodes.cpu().detach().numpy()
            nodes = add_salt_and_pepper_noise(nodes, noise_ratio)
            nodes = torch.from_numpy(nodes).float().to(device)

            output = self.model(nodes)

            loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for _, pearson, label, _ in dataloader:
            label = label.long()

            pearson, label = pearson.to(device), label.to(device)

            output = self.model(pearson)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        # auc = roc_auc_score(labels, result)
        # result = np.array(result)
        # result[result > 0.5] = 1
        # result[result <= 0.5] = 0
        # metric = precision_recall_fscore_support(
        #     labels, result, average='micro')
        # return [auc] + list(metric)

        auc = roc_auc_score(labels, result)
        fpr, tpr, thresholds = roc_curve(labels, result)  # 计算ROC曲线
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        # metric = precision_recall_fscore_support(
        #     labels, result,
        #     average='micro')  # precision_recall_fscore_support 直接把precision、recall、fscore和support的值都显示出来，其中support是每个标签在y_true中出现的次数
        acc = accuracy_score(labels, result)  # 计算准确率Accuracy
        sen = recall_score(labels, result)  # 计算灵敏度Sensitivity，即真阳性率True Positive Rate (TPR)
        ppv = precision_score(labels, result)  # 计算精确率Positive Predictive Value (PPV)，即查准率
        f_score = f1_score(labels, result)  # 计算F1分数
        mcc = matthews_corrcoef(labels, result)  # 计算MCC
        # 特异性需要使用混淆矩阵计算
        tn, fp, fn, tp = confusion_matrix(labels, result).ravel()
        spe = recall_score(labels, result, pos_label=0)  # 计算特异度Specificity，即真阴性率True Negative Rate (TNR)
        # return [auc] + list(metric)
        return [auc, fpr, tpr, thresholds, acc, sen, ppv, f_score, mcc, spe, tn, fp, fn, tp]


class FCNetTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder):
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.generated_graph = []

    def train_per_epoch(self, optimizer):

        self.model.train()

        for seq_group, label in self.train_dataloader:
            label = label.long()

            seq_group, label = seq_group.to(device), label.to(device)

            output = self.model(seq_group)

            loss = self.loss_fn(output, label)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_per_epoch(self, dataloader, loss_meter, acc_meter, save_graph=False):

        self.model.eval()

        self.generated_graph = []

        for seq_group, label in dataloader:
            label = label.long()

            seq_group, label = seq_group.to(device), label.to(device)

            output = self.model(seq_group)

            loss = self.loss_fn(output, label)

            loss_meter.update_with_weight(
                loss.item(), label.shape[0])

        return None

    def train(self):
        training_process = []
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])
            self.test_per_epoch(self.val_dataloader,
                                self.val_loss, self.val_accuracy)

            self.test_per_epoch(self.test_dataloader,
                                self.test_loss, self.test_accuracy, save_graph=True)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Edges:{self.edges_num.avg: .3f}',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%'
            ]))
            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg])

        self.save_result(training_process)
