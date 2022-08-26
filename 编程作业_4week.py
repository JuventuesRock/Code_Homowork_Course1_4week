# -*- coding: utf-8 -*-
# @Time : 2022/8/25 15:32
# @Author : zhuyu
# @File : 编程作业_4week.py
# @Project : Python菜鸟教程

"""
分别构建一个两层神经网络和多层(可选)神经网络
步骤：
1 初始化网络参数
2 前向传播
    2.1 计算一层的中线性求和的部分
    2.2 计算激活函数的部分（ReLU使用L-1次，Sigmod使用1次）
    2.3 结合线性求和与激活函数
3 计算误差
4 反向传播
    4.1 线性部分的反向传播公式
    4.2 激活函数部分的反向传播公式
    4.3 结合线性部分与激活函数的反向传播公式
5 更新参数
请注意，对于每个前向函数，都有一个相应的后向函数。 这就是为什么在我们的转发模块的每一步都会在cache中存储一些值，cache的值对计算梯度很有用， 在反向传播模块中，我们将使用cache来计算梯度。 现在我们正式开始分别构建两层神经网络和多层神经网络
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid,sigmoid_backward,relu,relu_backward
import lr_utils

#指定随机数种子
np.random.seed(1)

def print_test_infer(func_string):
    print("="*20+"测试"+func_string+"="*20)

#初始化参数
def initialize_parameters(n_x,n_h,n_y):
    """
    初始化两层神经网络参数
    :param n_x: 输入层节点数/输入特征数
    :param n_y: 隐藏层节点数
    :param n_z: 输出层节点数
    :return: parameters - 初始化后的参数字典
            W1 - 权重矩阵 维度(n_h,n_x)
            b1 - 偏向量 维度(n_h,1)
            W2 - 权重矩阵 维度(n_y,n_h)
            b2 - 偏向量 维度(n_y,1)
    """

    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros(shape=(n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros(shape=(n_y,1))

    assert W1.shape == (n_h,n_x)
    assert b1.shape == (n_h,1)
    assert W2.shape == (n_y,n_h)
    assert b2.shape == (n_y,1)

    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }

    return parameters

#测试initialize_parameters()
print_test_infer("initialize_parameters()")
parameters_test=initialize_parameters(3,2,1)
print(parameters_test)

#初始化深层神经网络
def initialize_parameters_deep(layers_dim):
    """
    初始化深层神经网络
    :param layers_dim:包含网络中各层节点数量的数组
    :return: parameters 包含网络各层参数的字典
            Wl - l层权重 维度(layer_num[l],layer_num[l-1])
            bl - l层偏向量 维度(layer_num[l],1)
    """

    np.random.seed(3)
    nums_layer=len(layers_dim) #网络的层数
    parameters = {} #创建一个存放参数的空字典

    for i in range(1,nums_layer):
        parameters["W"+str(i)]=np.random.randn(layers_dim[i],layers_dim[i-1])/np.sqrt(layers_dim[i-1])
        parameters["b"+str(i)]=np.zeros(shape=(layers_dim[i],1))

    return parameters

#测试initialize_parameters_deep()
print_test_infer("initialize_parameters_deep()")
layers_dim_test=[5,4,3]
parameters_test=initialize_parameters_deep(layers_dim_test)
print(parameters_test)

#前向传播包括线性部分和非线性部分
#线性部分
def linear_forward(A,W,b):
    """
    实现前向传播的线性部分
    :param A: 上一层的非线性输出A,维度(上一层的节点数量,样本数)
    :param W: 权重矩阵 维度(当前层节点数量,上一层节点数量)
    :param b: 偏置向量 维度(当前层节点数量,样本数量)
    :return: Z - 线性输出Z，作为激活函数的输入
            cache - 缓存"A","w","b"的字典，方便进行反向传播
    """
    Z=np.dot(W,A)+b
    assert Z.shape == (W.shape[0],A.shape[1])
    cache=(A,W,b)

    return Z,cache #以元组的形式返回(Z,cache)

#测试linear_forward()
print_test_infer("linear_forward()")
A_test,W_test,b_test=testCases.linear_forward_test_case()
Z_test,linear_cache_test=linear_forward(A_test,W_test,b_test)
print("Z = ",Z_test)

#线性激活部分
def linear_activation_forward(A_prey,W,b,activation="relu"):
    """
    实现Linear -> Activation 线性激活Z->A的函数
    :param A_prey: 上一层激活函数的输出 维度(上一层的节点数量,样本数)
    :param W: 权重矩阵 维度(当前层节点数量,上一层的节点数量)
    :param b: 偏置向量 维度(当前层节点数量,1)
    :param activation: 选择在此层中使用的激活函数名，字符串类型"sigmoid" or "relu"
    :return: A - 激活函数的输出
            cache - 包含”linear_cache“和”activation_cache“的字典，方便进行反向传播
            注意：这里我觉得没有必要包含activation_cache（缓存的是线性输出Z，Z已经在linear_cache中缓存过了)）
    """

    Z,linear_cache=linear_forward(A_prey,W,b)
    if activation == "sigmoid":
        A,activation_cache=sigmoid(Z)
    elif activation == "relu":
        A,activation_cache=relu(Z)

    assert A.shape == (W.shape[0],A_prey.shape[1])
    cache = (linear_cache,activation_cache)

    return A,cache

#测试linear_activation_forward()
print_test_infer("linear_activation_forward()")
A_prey_test,W_test,b_test = testCases.linear_activation_forward_test_case()
A_test,linear_activation_cache_test = linear_activation_forward(A_prey_test,W_test,b_test,activation="sigmoid")
print("sigmoid. A = ",A_test)
A_test,linear_activation_cache_test = linear_activation_forward(A_prey_test,W_test,b_test,activation="relu")
print("relu. A = ",A_test)

#多层(L)模型的前向传播
def L_model_forward(X,parameters):
    """
    实现多层神经网络的前向传播 前L-1层循环使用relu激活，最后一层使用sigmoid激活
    :param X:数据 维度(输入节点数量，样本数)
    :param parameters:模型参数 initialize_paramters_deep的输出
    :return: AL - 模型最后的输出值 最后一层激活函数的输出值
            caches - 包含以下内容的缓存列表
                linear_relu_forward()的每个cache（L-1个）
                linear_sigmoid_forward()的每个cache(1个)
    """
    A = X #输入X即A0
    # print(len(parameters))
    nums_layer=len(parameters)//2 #模型的层数
    # print(nums_layer)
    caches=[]
    for l in range(1,nums_layer):
        # print(l)
        A_prey= A
        A, cache = linear_activation_forward(A_prey,W=parameters["W"+str(l)],b=parameters["b"+str(l)],activation="relu")
        caches.append(cache)

    # print("out of for_loop l= ",l)
    AL,cache=linear_activation_forward(A,parameters["W"+str(nums_layer)],b=parameters["b"+str(nums_layer)],activation="sigmoid")
    caches.append(cache)

    assert (AL.shape == (1,X.shape[1])) #1应该是最后输出层的节点数量

    return AL,caches


#测试L_model_forward
print_test_infer("L_model_forward")
X_test,parameters_test = testCases.L_model_forward_test_case()
AL_test,caches_test=L_model_forward(X_test,parameters_test)
print("AL = ",AL_test)
print("caches 的长度为 = "+str(len(caches_test)))

#计算成本
def compute_cost(AL,Y):
    """
    计算成本函数 交叉熵损失函数公式参考视频
    :param AL:与标签预测相对应的概率向量，维度(1，样本数) 最后输出层一个节点即二分类判断是or不是
    :param Y: 标签向量，维度(1,样本数) 如果不是猫则为0 是猫则为1
    :return: cost - 交叉熵成本
    """

    m=Y.shape[1]
    cost=-np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),(1-Y)))/m

    cost=np.squeeze(cost)
    assert cost.shape == ()

    return cost


#测试compute_cost
print_test_infer("compute_cost")
Y_test,AL_test=testCases.compute_cost_test_case()
cost_test=compute_cost(AL_test,Y_test)
print("cost_test = ",cost_test)

#反向传播
# 与前向传播类似，我们有需要使用三个步骤来构建反向传播：
# LINEAR 后向计算
# LINEAR -> ACTIVATION 后向计算，其中ACTIVATION 计算Relu或者Sigmoid 的结果
# [LINEAR -> RELU] × \times× (L-1) -> LINEAR -> SIGMOID 后向计算 (整个模型)

#线性部分
def linear_backward(dZ,cache):
    """
    为单层实现反向传播的线性部分(第L层)
    :param dZ: 相对于该层线性输出的梯度
    :param cache: 来自当前层前向传播时的缓存(A,W,b)
    :return:
            dA_prey - 相对于上一层激活函数的成本梯度，与A_prey的维度相同
            dW - 相对于当前层W的成本梯度，与W的维度相同
            db - 相对于当前层b的成本梯度，与b的梯度相同
    """

    A_prey,W,b=cache
    m = A_prey.shape[1]
    dW=np.dot(dZ,A_prey.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prey=np.dot(W.T,dZ)

    assert dA_prey.shape==A_prey.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prey,dW,db

#测试linear_forward
print("linear_forward")
dZ_test,caches_test=testCases.linear_backward_test_case()
dA_prey_test,dW_test,db_test=linear_backward(dZ_test,caches_test)
print("dA_prey = ",dA_prey_test)
print("dW = ",dW_test)
print("db = ",db_test)

#线性激活部分
# 为了帮助你实现linear_activation_backward，我们提供了两个后向函数
#sigmoid_backward:实现了sigmoid（）函数的反向传播 计算dZ
#relu_backward: 实现了relu（）函数的反向传播 计算dZ

def linear_activation_backward(dA,cache,activation="relu"):
    """
    实现Linear -> Activation的后向传播 复杂点在于激活函数dA对于线性输出dZ的导数需要分别推导求出，这里已经提前封装好激活函数对应的dZ了
    :param dA: 当前层l激活后的梯度值
    :param cache: 当前层前向传播时的缓冲值 (linear_cache,activation_cache)
    :param acitivation:激活函数的选择
    :return:
            dA_prey - 相对于前一层(l-1)激活函数的成本梯度 维度与A_prey相同
            dW - 相对于当前层W的成本梯度，维度与W相同
            db - 相对于当前层b的成本梯度，维度与b相同
    """
    linear_cache,activation_cache=cache
    #不同的activation对应的dZ不同，即用于linear_backward的输入不同
    if activation == "relu":
        dZ=relu_backward(dA,activation_cache)

    elif activation =="sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)

    dA_prey, dW, db = linear_backward(dZ, cache=linear_cache) #为什么可以访问到函数内dZ呢？查一查


    return dA_prey,dW,db

#
# def linear_activation_backward(dA, cache, activation="relu"):
#     """
#     实现LINEAR-> ACTIVATION层的后向传播。
#
#     参数：
#          dA - 当前层l的激活后的梯度值
#          cache - 我们存储的用于有效计算反向传播的值的元组（值为linear_cache，activation_cache）
#          activation - 要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
#     返回：
#          dA_prev - 相对于激活（前一层l-1）的成本梯度值，与A_prev维度相同
#          dW - 相对于W（当前层l）的成本梯度值，与W的维度相同
#          db - 相对于b（当前层l）的成本梯度值，与b的维度相同
#     """
#     linear_cache, activation_cache = cache
#     if activation == "relu":
#         dZ = relu_backward(dA, activation_cache)
#         dA_prev, dW, db = linear_backward(dZ, linear_cache)
#     elif activation == "sigmoid":
#         dZ = sigmoid_backward(dA, activation_cache)
#         dA_prev, dW, db = linear_backward(dZ, linear_cache)
#
#     return dA_prev, dW, db


#测试linear_activation_backward()
print("linear_activation_backward")
AL_test, linear_activation_cache_test = testCases.linear_activation_backward_test_case()

dA_prev_test, dW_test, db_test = linear_activation_backward(AL_test, linear_activation_cache_test, activation="sigmoid")
print("sigmoid:")
print("dA_prev = " + str(dA_prev_test))
print("dW = " + str(dW_test))
print("db = " + str(db_test) + "\n")

dA_prev_test, dW_test, db_test = linear_activation_backward(AL_test, linear_activation_cache_test, activation="relu")
print("relu:")
print("dA_prev = " + str(dA_prev_test))
print("dW = " + str(dW_test))
print("db = " + str(db_test))

#多层神经网络的反向传播
def L_model_backward(AL,Y,caches):
    """
    多层神经网络的反向传播
    :param AL: 反向传播的起始输入值AL(即正向传播的输出 - L_model_forward)
    :param Y: 标签向量 维度(1,样本数)
    :param caches: 包含以下内容的caches列表
            linear_activation_forward(relu)的cache 不包含输出层
            linear_activation_for(sigmoid)的cache 输出层
    :return:
        grads - 具有各层梯度的字典
            grad["dA"+str(l)]=
            grad["dw"+str(l)]=
            grad["db"+str(l)]=
    """

    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)
    dAL=-(np.divide(Y,AL)-np.divide(1-Y,1-AL)) #计算反向传播的起始值

    current_cache=caches[L-1]
    grads["dA"+str(L)],grads["dW"+str(L)],grads["db"+str(L)]=linear_activation_backward(dAL,current_cache,activation="sigmoid")

    for l in reversed(range(L-1)):
        current_cache=caches[l]
        grads["dA"+str(l+1)],grads["dW"+str(l+1)],grads["db"+str(l+1)]=linear_activation_backward(grads["dA"+str(l+2)],current_cache,activation="relu")

    return grads

#测试L_model_backward()
print_test_infer("L_model_backward")
AL_test,Y_test,caches_test=testCases.L_model_backward_test_case()
grads_test=L_model_backward(AL_test,Y_test,caches_test)
print("dW1 = ",grads_test["dW1"])
print("db1 = ",grads_test["db1"])
print("dA1 = ",grads_test["dA1"])

#更新参数
def update_parameters(paramters,grads,learning_rate=0.01):
    """
    使用梯度下降更新参数
    :param paramters:包含模型参数的字典
    :param grads: 包含模型各层参数梯度的字典
    :param learning_rate: 学习率
    :return: paramters - 更新后的参数字典
    """

    L=len(paramters)//2 #模型层数
    for l in range(L):
        paramters["W"+str(l+1)]=paramters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        paramters["b"+str(l+1)]=paramters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]

    return paramters

#测试update_parameters
print_test_infer("update_paramters")
parameters_test,grads_test=testCases.update_parameters_test_case()
parameters_test=update_parameters(parameters_test,grads_test,0.1)

print ("W1 = "+ str(parameters_test["W1"]))
print ("b1 = "+ str(parameters_test["b1"]))
print ("W2 = "+ str(parameters_test["W2"]))
print ("b2 = "+ str(parameters_test["b2"]))

#将上述方法组合起来搭建神经网络  分别搭建一个两层神经网络和多层神经网络
#两层神经网络
def two_layers_model(X,Y,layer_dims,learning_rate=0.075,num_iteration=3000,print_cost=False,isPlot=True):
    """
    实现一个两层的神经网络
    :param X: 输入的数据 维度(n_x,样本数)
    :param Y: 标签向量 维度(1,样本数)
    :param layer_dims: 包含各层节点数量的元组，维度为(nx,nh,ny)
    :param learning_rate: 学习率
    :param num_iteration: 迭代的次数
    :param print_cost: 是否打印cost
    :param isPlot: 是否绘制出cost的变化图
    :return:
        :parameters - 一个包含模型参数W1 b1 W2 b2的字典变量
    """
    np.random.seed(1)
    grads={}
    costs=[]
    (n_x,n_h,n_y)=layer_dims

    """
    初始化模型参数
    """
    parameters=initialize_parameters(n_x,n_h,n_y)
    W1,b1 =parameters["W1"],parameters["b1"]
    W2,b2 =parameters["W2"],parameters["b2"]

    """
    开始进行迭代
    """
    for i in range(0,num_iteration):
        #前向传播
        A1,cache1=linear_activation_forward(X,W1,b1,activation="relu")
        A2,cache2=linear_activation_forward(A1,W2,b2,activation="sigmoid")

        #计算损失
        cost=compute_cost(A2,Y)

        #反向传播
        #初始化反向传播
        dA2=-(np.divide(Y,A2)-np.divide(1-Y,1-A2))
        dA1,dW2,db2=linear_activation_backward(dA2,cache2,activation="sigmoid"  )
        dA0,dW1,db1=linear_activation_backward(dA1,cache1,activation="relu"  ) #dA0即对X的成本梯度，没有意义不需要

        #保存梯度
        grads["dW1"]=dW1
        grads["db1"]=db1
        grads["dW2"]=dW2
        grads["db2"]=db2

        #更新参数
        parameters=update_parameters(parameters,grads=grads,learning_rate=learning_rate)
        W1,b1=parameters["W1"],parameters["b1"]
        W2,b2=parameters["W2"],parameters["b2"]

        if i%100 == 0 :
            #记录成本
            costs.append(cost)
            #是否打印成本
            if print_cost:
                print("第{0}次迭代,cost = {1}".format(i,np.squeeze(cost)))

    #迭代完成后，根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iteration (per tens)")
        plt.title("Learning rate = "+str(learning_rate))
        plt.show()

    return parameters

#加载并处理图像数据集
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=lr_utils.load_dataset()

train_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_x=train_x_flatten/255 #归一化
train_y=train_set_y
test_x=test_x_flatten/255
test_y=test_set_y

#开始训练 二层神经网络
n_x,n_y,n_z=(12288,7,1)
layers_dim=(n_x,n_y,n_z)
parameters=two_layers_model(train_x,train_y,layers_dim,learning_rate=0.0075,num_iteration=2500,print_cost=True,isPlot=True)

#预测
def predict(X,y,parameters):
    """
    该函数用于预测L层神经网络的效果 也包括二层神经网络
    :param X: 输入测试数据 维度(n_x,样本数)
    :param y: 标签向量 维度(1,样本数)
    :param parameter: 模型训练后的参数字典 parameters
    :return: p - 给定数据集X的预测accuracy
    """

    m=X.shape[1]
    n=len(parameters)//2 #神经网络的层数
    p=np.zeros((1,m))

    #根据参数进行前向传播
    probas,caches=L_model_forward(X,parameters)

    for i in range(0,probas.shape[1]):
        if probas[0,i]>0.5:
            p[0,i]=1
        else:
            p[0,i]=0

    print("准确度accuracy = {}%".format(float(np.sum(p==y)/m)*100))

    return p

#开始预测 分别查看训练集和测试集预测的accuracy
prediction_train=predict(train_x,train_y,parameters)
prediction_test=predict(test_x,test_y,parameters)

#搭建多层神经网络
def L_layer_model(X,Y,layer_dims,learning_rate=0.0075,num_iterations=3000,print_cost=True,isPlot=True):
    """
    实现L层神经网络
    :param X: 输入数据
    :param Y: 标签向量
    :param layer_dims: 包含L层各层的节点数量的元组/列表
    :param learning_rate: 学习率
    :param num_iterations: 迭代的次数
    :param print_cost: 是否打印成本
    :param isPlot: 是否绘制成本变化曲线
    :return: 包含L层模型各层参数的字典变量 parameters
    """
    np.random.seed(1)
    costs=[]

    parameters=initialize_parameters_deep(layers_dim)

    #前向传播
    for i in range(num_iterations):
        AL,caches=L_model_forward(X,parameters)

        cost=compute_cost(AL,Y)

        grads=L_model_backward(AL,Y,caches)

        parameters=update_parameters(parameters,grads,learning_rate=learning_rate)

        if i %100 ==0 :
            costs.append(cost)
            if print_cost:
                print("第{0}次迭代，cost = {1}".format(i,np.squeeze(cost)))

    #迭代完成
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iteration (per ten)")
        plt.title("Learning rate = "+str(learning_rate))
        plt.show()

    return parameters

#加载数据集 (和之前的一样)
train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

#加载完成，开始训练多层神经网络
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True,isPlot=True)

#开始预测 （多层神经网络在训练集和测试集上分别预测）
prediction_train=predict(train_x,train_y,parameters)
prediction_test=predict(test_x,test_y,parameters)


#看一看有哪些东西在L层模型中被错误地标记了，导致准确率没有提高。（答案中的函数）
def print_mislabeled_images(classes, X, y, p):
    """
	绘制预测和实际不同的图像。
	    X - 数据集
	    y - 实际的标签
	    p - 预测
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))
    plt.show()


print_mislabeled_images(classes, test_x, test_y, prediction_test)


