import deepdish as dd
import os.path as osp
import os
import numpy as np
import argparse
from pathlib import Path
import pandas as pd


def main(args):
    data_dir = os.path.join(args.root_path, 'data/raw')
    timeseires = os.path.join(args.root_path, 'data')

    meta_file = os.path.join(args.root_path, 'data/participants.tsv')

    meta_file = pd.read_csv(meta_file, header=0, delimiter='\t')  # header=0指定第几行作为列名

    # id2site = meta_file[["subject", "SITE_ID"]]
    #
    # # pandas to map
    # id2site = id2site.set_index("subject")  # set_index可以指定数据中的某一列，将其作为该数据的新索引
    # id2site = id2site.to_dict()['SITE_ID']  # to_dict()方法将列名设置为字典键

    times = []

    labels = []
    pcorrs = []

    corrs = []
    id_list = []
    # site_list = []

    for f in os.listdir(data_dir):  # 获取指定文件夹下的所有文件
        if osp.isfile(osp.join(data_dir, f)):  # 用于判断对象是否为一个文件
            if '_' in f:
                fname = f.split('_')[0]
                # site = id2site[int(fname)]

                files = os.listdir(osp.join(timeseires, fname))

                # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
                # Lambda 函数又称匿名函数，即用句子实现函数的功能
                # 即过滤出files中以1D后缀的元素
                for i in range(3):
                    file = list(filter(lambda x: x.endswith("txt"), files))[i]
                    print(file)

                    time = np.loadtxt(osp.join(timeseires, fname, file), skiprows=0).T  # skiprows=n：指跳过前n行

                    if time.shape[1] < 100:
                        continue

                    fl = os.path.join(data_dir, fname + "_" + str(i) + ".h5")
                    print(fl)

                    temp = dd.io.load(fl)
                    pcorr = temp['pcorr'][()]

                    pcorr[pcorr == float('inf')] = 0  # float('inf')表示正负无穷

                    att = temp['corr'][()]

                    att[att == float('inf')] = 0  # 看输出结果是对角线置0

                    label = temp['label']

                    times.append(time[:, :100])  # 200*100
                    labels.append(label[0])
                    corrs.append(att)
                    pcorrs.append(pcorr)
                    # site_list.append(site)
                    id_list.append(int(fname))

    print(len(times))
    print(len(labels))
    print(len(corrs))
    print(len(pcorrs))
    print(len(id_list))
    # 所有受试者的所有信息保存至 ABIDE_pcp/abide.npy
    np.save(Path(args.root_path) / 'rd_td.npy',
            {'timeseires': np.array(times), "label": np.array(labels), "corr": np.array(corrs),
             "pcorr": np.array(pcorrs), 'id': np.array(id_list)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the final dataset')
    parser.add_argument('--root_path', default="/mnt/5468e/ypzhang/LiYuxin/RD-TD", type=str,
                        help='The path of the folder containing the dataset folder.')
    args = parser.parse_args()
    main(args)