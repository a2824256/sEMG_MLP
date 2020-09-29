from paddle import fluid
import scipy.io as scio
import glob
import numpy as np

# 数据集路径
path = "./Data/"
# 文件列表
img_list = glob.glob(path + "*.mat")
train_set = None
test_set = None

def train_sample_reader():
    for i in range(100):
        feature_1 = np.array(train_set[:, 0]).astype('float32')
        feature_2 = np.array(train_set[:, 1]).astype('float32')
        feature_3 = np.array(train_set[:, 2]).astype('float32')
        label = np.array(train_set[:, 4]).astype('int64')
        yield feature_1, feature_2, feature_3, label


def test_sample_reader():
    for i in range(10):
        feature_1 = np.array(train_set[:, 0]).astype('float32')
        feature_2 = np.array(train_set[:, 1]).astype('float32')
        feature_3 = np.array(train_set[:, 2]).astype('float32')
        label = np.array(train_set[:, 4]).astype('int64')
        yield feature_1, feature_2, feature_3, label


print("读取文件:")
for file_path in img_list:
    print(file_path)
    data_dic = scio.loadmat(file_path)
    if train_set is None:
        train_set = data_dic['FeatureSet']
    else:
        train_set = np.concatenate((train_set, data_dic['FeatureSet']), axis=0)
print("合并后的数据集尺寸：")
print(train_set.shape)
