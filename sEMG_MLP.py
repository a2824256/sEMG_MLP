from paddle import fluid
import scipy.io as scio
import glob
import numpy as np
import sys, math
from paddle.utils.plot import Ploter
# 数据集路径
path = "DATA/"
# 数据列表
input_1_list = []
input_2_list = []
input_3_list = []
label_list = []
# 文件列表
img_list = glob.glob(path + "feature*.mat")
train_set = None
test_set = None

def train_sample_reader():
    for i in range(10):
        feature_1 = np.array(input_1_list).astype('float64')
        feature_2 = np.array(input_2_list).astype('float64')
        feature_3 = np.array(input_3_list).astype('float64')
        label = np.array(label_list).astype('int64')

        yield feature_1, feature_2, feature_3, label


def test_sample_reader():
    for i in range(10):
        feature_1 = np.array(input_1_list).astype('float64')
        feature_2 = np.array(input_2_list).astype('float64')
        feature_3 = np.array(input_3_list).astype('float64')
        label = np.array(label_list).astype('int64')
        yield feature_1, feature_2, feature_3, label


def train(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(program=program,
                            feed=feeder.feed(data_test),
                            fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]  # 累加测试过程中的损失值
        count += 1  # 累加测试集中的样本数量
    return [x_d / count for x_d in accumulated]

print("读取文件:")
# data pretreatment
for file_path in img_list:
    data_dic = scio.loadmat(file_path)
    if train_set is None:
        train_set = data_dic['FeatureSet']
    else:
        train_set = np.concatenate((train_set, data_dic['FeatureSet']), axis=0)

print("合并后的数据集尺寸：")
train_set = train_set.tolist()
for row in range(len(train_set)):
    input_1_list.append(train_set[row][0][0].tolist())
    input_2_list.append(train_set[row][1][0].tolist())
    input_3_list.append(train_set[row][2][0].tolist())
    label_list.append([train_set[row][4][0][0]-1])
input_3_list = np.array(input_3_list).astype('float64').tolist()
train_reader = fluid.io.batch(train_sample_reader, batch_size=1)
test_reader = fluid.io.batch(test_sample_reader, batch_size=1)

print("network")
# network
input_1 = fluid.data(name='input_1', shape=[None, 8], dtype='float64')
input_2 = fluid.data(name='input_2', shape=[None, 8], dtype='float64')
input_3 = fluid.data(name='input_3', shape=[None, 8], dtype='float64')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')
hidden_1 = fluid.layers.fc(name='fc1', input=input_1, size=60, act='relu')
hidden_2 = fluid.layers.fc(name='fc2', input=input_2, size=30, act='relu')
hidden_3 = fluid.layers.fc(name='fc3', input=input_3, size=20, act='relu')
prediction = fluid.layers.fc(name='pred', input=[hidden_1, hidden_2, hidden_3], size=10, act='softmax')

print("main program")
# main program
main_program = fluid.default_main_program()  # 获取默认/全局主函数
startup_program = fluid.default_startup_program()

# loss
loss = fluid.layers.mean(fluid.layers.cross_entropy(input=prediction, label=label))
acc = fluid.layers.accuracy(input=prediction, label=label)

# test program
test_program = main_program.clone(for_test=True)
adam = fluid.optimizer.Adam(learning_rate=0.01)
adam.minimize(loss)

# place = fluid.CPUPlace()
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)

num_epochs = 10

params_dirname = "./my_paddle_model"

feeder = fluid.DataFeeder(feed_list=[input_1, input_2, input_3, label], place=place)
exe.run(startup_program)
train_prompt = "train cost"
test_prompt = "test cost"

# %matplotlib inline
plot_prompt = Ploter(train_prompt, test_prompt)
step = 0

exe_test = fluid.Executor(place)
print("start training")
for pass_id in range(num_epochs):
    for data_train in train_reader():
        step = step + 1
        avg_loss_value, = exe.run(main_program, feed=feeder.feed(data_train), fetch_list=[loss])
        if step % 10 == 0:  # 每10个批次记录并输出一下训练损失
            # plot_prompt.append(train_prompt, step, avg_loss_value[0])
            # plot_prompt.plot()
            print("%s, Step %d, Cost %f" % (train_prompt, step, avg_loss_value[0]))
        if step % 10 == 0:  # 每100批次记录并输出一下测试损失
            test_metics = train(executor=exe_test, program=test_program, reader=test_reader, fetch_list=[loss.name], feeder=feeder)
            # plot_prompt.append(test_prompt, step, test_metics[0])
            # plot_prompt.plot()
            print("%s, Step %d, Cost %f" % (test_prompt, step, test_metics[0]))
            if test_metics[0] < 10.0:  # 如果准确率达到要求，则停止训练
                break

        if math.isnan(float(avg_loss_value[0])):
            sys.exit("got NaN loss, training failed.")

        #  保存训练参数到之前给定的路径中
        if params_dirname is not None:
            fluid.io.save_inference_model(params_dirname, ['input_1','input_2','input_3'], [prediction], exe)