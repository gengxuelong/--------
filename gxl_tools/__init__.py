
__all__ = ['gxl1','gxl2']#__init__.py的内容可以为空，一般用来进行包的某些初始化工作或者设置__all__值
#，__all__是在from package-name import *这语句使用的，全部导出定义过的模块。



import random
import time
import numpy as np
from matplotlib import pyplot as plt
import math
from torch.utils import data 
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from . import gxl
from IPython import display#这段代码使用了IPython的`display`模块中的`clear_output`函数，该函数的作用是清除当前输出区域的内容。其中，`wait=True`参数指定在清除输出前等待新的输出。

def test_showimage_function():
    print('hello gengxuelong')
    mnist_train = torchvision.datasets.FashionMNIST(root='../data',transform=transforms.ToTensor(),train=True,download=True)
    X,y = next(iter(data.DataLoader(mnist_train, batch_size=10)))
    show_image_for_arrays(X.reshape(10,28,28),y,5)

def hellogxl():
    print('hello 耿雪龙')


def get_fashion_mnist_labels(labels):
    """返回标签的文本标签列表"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


class Timer:
    '''龙氏计时器'''
    def __init__(self) -> None:
        self.times = []
        self.tik = None
        self.start()
    def start(self):
        self.tik = time.time()
    def end(self):
        gap = time.time()-self.tik
        self.times.append(gap)
        return self.times[-1]
    def avg(self):
        n = len(self.times)
        return sum(self.times)/n
    def sum(self):
        return sum(self.times)
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
    
def normal_equation(x,mu,sigma):
    """龙氏正态分布方程"""
    parameter = 1/math.sqrt(2*math.pi*sigma**2)
    right = np.exp(-1/(2*sigma**2)*(x-mu)**2)
    return parameter*right 

def init_data_normal(mu,sigma,size=(10,2)):
    """得到指定size的正态分布随机变量,得到numpy array"""
    # random 0-1 的mu=0.5 sigma=math.sqrt(1/12)
    X = np.random.random(size=size)
    X = (X - 0.5)/math.sqrt(1/12)
    X = X*sigma+mu
    return X

def init_data_normal_for_torch(mu,sigma,size=(10,2)):
    """得到指定size的正态分布随机变量,for torch,得到记录梯度的tensor"""
    X = init_data_normal(mu,sigma,size)
    return torch.tensor(X,requires_grad=True)

def init_data_ones(size=(10)):
    """得到指定size的ones,得到numpy array"""
    # random 0-1 的mu=0.5 sigma=math.sqrt(1/12)
    X = np.ones(shape=size)
    return X

def init_data_ones_for_torch(size=(10)):
    """得到指定size的ones,for torch,得到记录梯度的tensor"""
    X = init_data_ones(size)
    return torch.tensor(X,requires_grad=True)

def init_data_zeros(size=(10)):
    """得到指定size的zeros,得到numpy array"""
    # random 0-1 的mu=0.5 sigma=math.sqrt(1/12)
    X = np.zeros(shape=size)
    return X

def init_data_zeros_for_torch(size=(10)):
    """得到指定size的zeros,for torch,得到记录梯度的tensor"""
    X = init_data_zeros(size)
    return torch.tensor(X,requires_grad=True)


def make_data_by_parameters_for_linear(w,b,nums):
    """通过w 和 b得到若干随机的点对"""
    X = torch.normal(0,1,(nums,len(w)))
    Y = torch.matmul(X,w)+b
    Y += torch.normal(0,0.01,Y.shape)
    return X,Y.reshape(-1,1)

def data_iter_getter(batch_size, features, labels):
    """通过特征和标签得到数据迭代器"""
    num = len(features)
    indices = list(range(num))
    random.shuffle(indices)
    for i in range(0,num,batch_size):
        indices_batch = torch.tensor(indices[i:min(i+batch_size,num)])
        yield features[indices_batch],labels[indices_batch]

def line_mode(W,b):
    '''龙氏线性模型'''
    W=W.type(torch.float32)
    b=b.type(torch.float32)
    return lambda X : (torch.matmul(X.type(torch.float32).reshape(-1,len(W)),W)+b)

def squared_loss(y_hat,y,action='none'):
    '''龙氏平方损失函数,action参数决定动作'''
    temp = (y_hat-y.reshape(y_hat.size()))**2/2
    if(action == 'mean'):
        return temp.mean()
    if(action == 'sum'):
        return temp.sum()
    return temp
    
def sgd(params,lr:int):
    '''龙氏简单梯度优化器'''
    def anonymity():
        with torch.no_grad():
            for param in params:
                param -= lr*param.grad #。当我们计算的损失是⼀个批量样本的总和时，我们⽤批量⼤⼩（batch_size）来规范化步⻓，这样步⻓⼤⼩就不会取决于我们对批量⼤⼩的选择
                param.grad.zero_()
    return anonymity


def get_dataloader_nums()->int:
    """设置加载数据时的线程数"""
    return 4


def load_data_iter_by_dataarrays_by_frame(data_arrays,batch_size:int=20,is_train=True,num_workers=get_dataloader_nums()):
    """通过框架内置方法,通过data_arrays得到数据的迭代器,第一个参数形如:(features,labels)"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train,num_workers=num_workers)

def load_data_iter_by_dataset_by_frame(dataset,batch_size:int=20,is_train=True,num_workers=get_dataloader_nums()):
    """通过框架内置方法,通过dataset对象得到数据的迭代器"""
    return data.DataLoader(dataset,batch_size,shuffle=is_train,num_workers=num_workers)

def get_next_from_iter(data_iter):
    """得到iter的下一个输出"""
    return next(iter(data_iter))


def show_image_for_arrays(image_arrays,text_labels,col_num=5):
    """展示数组形式的图片"""
    num = len(image_arrays)
    row_num = math.ceil(num/col_num)
    plt.figure(figsize=(20,6*row_num))
    for i in range(1,num+1):
        plt.subplot(row_num,col_num,i)
        plt.imshow(image_arrays[i-1])
        plt.xticks([])
        plt.yticks([])
        plt.title(text_labels[i-1])
    plt.subplots_adjust(hspace =0,wspace=0)#调整子图间距
    plt.show()


def load_data_iter_of_fashion_mnist(batch_size:int=50,resize=None):
    """下载mnist数据集,并将其以data_iter的形式加载到内存中,resize是改变图片的形状"""
    trains = [transforms.ToTensor()]
    if(resize):
        trains.insert(0,transforms.Resize(resize))
    trains = transforms.Compose(trains)
    mnist_train = torchvision.datasets.FashionMNIST(root='../data',train=True,transform=trains,download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data',train=False,transform=trains,download=True)
    # return data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_nums()),data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=get_dataloader_nums())
    return load_data_iter_by_dataset_by_frame(mnist_train,batch_size=batch_size,is_train=True),load_data_iter_by_dataset_by_frame(mnist_train,batch_size=batch_size,is_train=True)

def load_data_iter_by_dataset_by_myfunction(dataset,batch_size:int):
    """从dataset中得到数据迭代器,自定义方式"""
    num = len(dataset)
    indices = list(range(num))
    random.shuffle(indices)
    for i in range(0,num,batch_size):
        indices_batch = torch.tensor(indices[i:min(i+batch_size,num)])
        features_batch = []
        lables_batch = []
        for i in indices_batch:
            features_batch.append(dataset[i][0])
            lables_batch.append(dataset[i][1])
        yield features_batch,lables_batch

def softmax_equation(X):
    """龙氏softmax的方程"""
    exp_x = torch.exp(X)
    denominator = exp_x.sum(1,keepdim=True)
    return exp_x/denominator


def reLU_equation(X):
    """龙氏relu的方程"""
    zeros = torch.zeros_like(X)
    X = (X > zeros).float()
    return X

def softmax_mode(W,b):
    """龙氏softmax模型,自定义"""
    return lambda X : softmax_equation(line_mode(W,b)(X)) #匿名函数有个限制，就是只能有一个表达式，不写return，返回值就是该表达式

def cross_entropy_loss(y_hat,y,action='none'):
    """龙氏交叉熵,aciton参数决定动作"""
    temp = -torch.log(y_hat[range(len(y_hat)),y])
    if(action == 'mean'):
        return temp.mean()
    if(action == 'sum'):
        return temp.sum()
    return temp

def accuracy_for_right_number(y_hat,y):
    """得到正确的个数"""
    if(len(y_hat.shape)>1 and y_hat.shape[0]>1):
        y_hat = y_hat.argmax(axis=1)#返回每一行中最大元素的索引，也就是对每个样本预测的类别。其中，axis=1表示按行计算。具体来说，如果y_hat的shape为(n_samples, n_classes)，那么y_hat.argmax(axis=1)的shape就为(n_samples,)，其中每个元素是一个整数，表示对应样本的预测类别。
    one_hot_array = y_hat.type(y.dtype)==y
    return float(one_hot_array.type(y.dtype).sum())

class Accumulator():
    def __init__(self,n) -> None:
        self.data = [0.0]*n
    def add(self,*args):
        if(len(args)!=len(self.data)):
            print('error: sorry,the number of parameter is wrong,the function will not work')
            return
        self.data = [a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data = [0.0]*len(self.data)
    def __getitem__(self,index):
        return self.data[index]

def evaluate_accuracy_for_net(net,data_iter):
    """评估在测试集合上的正确数量"""
    if(isinstance(net,torch.nn.Module)):
        net.eval()#开启评估模式
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy_for_right_number(net(X),y),y.numel())
    return metric[0]/metric[1]

def evaluate_loss_for_net(net,loss,data_iter):
    """评估在测试集上的损失"""
    if(isinstance(net,torch.nn.Module)):
        net.eval()#开启评估模式
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            y_hat = net(X)
            l = loss(y_hat,y.reshape(y_hat.shape))
            metric.add(float(l.sum()),y.numel())
    return metric[0]/metric[1]

class Animator():
    def __init__(self,line_names=['train loss','test loss']) -> None:
        self.lines_dic = {}
        self.lines_name = line_names

    def add_dynamic(self,*args):
        display.clear_output(wait=True)
        if 'line0' not in self.lines_dic:
            self.lines_dic['line0'] = []
            (self.lines_dic['line0']).append(args[0])
        else:
            (self.lines_dic['line0']).append(args[0])
        for i in range(1,len(args)):
            if 'line'+str(i) not in self.lines_dic:
                self.lines_dic['line'+str(i)] = []
                (self.lines_dic['line'+str(i)]).append(args[i])
            else:
                (self.lines_dic['line'+str(i)]).append(args[i])
            plt.plot(self.lines_dic['line0'],self.lines_dic['line'+str(i)],"*-",label=self.lines_name[i-1])
        plt.xlabel('epochs')
        plt.ylabel('number')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
        plt.pause(0.1)

def start_train_for_one_epoch_for_classify(net,loss,optimizer,train_iter):
    """开始一个epoch的训练"""
    if(isinstance(net,torch.nn.Module)):
        net.train()# 开启训练模式
    metric = Accumulator(3)# 累计损失 , 累计错误数, 总数
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        if(isinstance(optimizer,torch.optim.Optimizer)):
            optimizer.zero_grad()
            l.mean().backward()
            optimizer.step()
        else:
            l.mean().backward()
            optimizer()
        metric.add(float(l.sum()),accuracy_for_right_number(y_hat,y),y.numel())
    # print(f'loss : {metric[0]/metric[2]} accuracy : {metric[1]/metric[2]}')
    return float(metric[0]/metric[2]),float(metric[1]/metric[2])

def start_train_for_one_epoch_for_regression(net,loss,optimizer,train_iter):
    """开始一个epoch的训练,返回训练损失均值"""
    if(isinstance(net,torch.nn.Module)):
        net.train()# 开启训练模式
    metric = Accumulator(2)# 累计损失 ,  总数
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        if(isinstance(optimizer,torch.optim.Optimizer)):
            optimizer.zero_grad()
            l.mean().backward()
            optimizer.step()
        else:
            l.mean().backward()
            optimizer()
        metric.add(float(l.sum()),y.numel())
    # print(f'one epoch: loss : {metric[0]/metric[2]} accuracy : {metric[1]/metric[2]}')
    return float(metric[0]/metric[1])


def train_for_some_epoch_for_classify(epoch_num,net,loss,optimizer,train_iter,test_iter):
    """开始全程训练,带动画展示过程"""
    animator = Animator(['train loss','train acc','test acc'])
    for epoch in range(1,epoch_num+1):
        train_loss ,train_acc = start_train_for_one_epoch_for_classify(net,loss,optimizer,train_iter)
        test_acc = evaluate_accuracy_for_net(net,test_iter)
        animator.add_dynamic(epoch,train_loss,train_acc,test_acc)

def train_for_some_epoch_for_regression(epoch_num,net,loss,optimizer,train_iter,test_iter):
    """开始全程训练,带动画展示过程"""
    animator = Animator(['train_loss','test_loss'])
    for epoch in range(1,epoch_num+1):
        test_loss = evaluate_loss_for_net(net,loss,test_iter)
        train_loss = start_train_for_one_epoch_for_regression(net,loss,optimizer,train_iter)
        animator.add_dynamic(epoch,train_loss,test_loss)


def predict_for_mnist_in_ch3(net,test_iter,n=6):
    """预测mnist数据集"""
    for X,y in test_iter:
        break
    print(X.shape)
    labels = get_fashion_mnist_labels(y)
    labels_hat = get_fashion_mnist_labels(net(X).argmax(axis=1))
    labels_gather = [true+"\n"+pred for true,pred in zip(labels,labels_hat)]
    show_image_for_arrays(X[:n].reshape(n,28,28),labels_gather)


def train_polynomial_example(train_dim=4,true_dim=4,the_true_w=[3.14,6.28,5,4],max_degree=20,train_num=100,test_num=100,batch_size=10,epochs=40,learn_rate=0.01):
    """多项式训练例子,直观展示正拟合,过拟合和欠拟合"""
    batch_size = min(batch_size,train_num)
    true_w= np.zeros(max_degree)
    true_w[:true_dim] = np.array(the_true_w)
    features = np.random.normal(size=(train_num+test_num,1))
    np.random.shuffle(features)
    poly_feattures = np.power(features,np.arange(max_degree).reshape(1,-1))
    for i in range(max_degree):
        poly_feattures[:,i]/=math.gamma(i+1) # gamma(n) = (n-1)!
    labels = np.dot(poly_feattures,true_w)
    labels += np.random.normal(scale=0.1,size=labels.shape)
    true_w,features,poly_feattures,labels = [torch.tensor(a,dtype=torch.float32) for a in [true_w,features,poly_feattures,labels]]
    # 训练
    train_iter = load_data_iter_by_dataarrays_by_frame((poly_feattures[:train_num,:train_dim],labels[:train_num].reshape(-1,1)),batch_size)#reshape(-1,1)变成二维矩阵,非常重要
    test_iter = load_data_iter_by_dataarrays_by_frame((poly_feattures[train_num:,:train_dim],labels[train_num:].reshape(-1,1)),batch_size,False)
    net = nn.Sequential(nn.Linear(train_dim,1,bias=False))
    def weights_init(m):
        if(type(m)==nn.Linear):
            m.weight.data.normal_(0,0.01)
    net.apply(weights_init)
    loss = nn.MSELoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(),lr = learn_rate)
    train_for_some_epoch_for_regression(epochs,net,loss,optimizer,train_iter,test_iter)
    print(net[0].weight.data)

def get_init_params(input_num,output,std=1,mean=0):
    """通过输入和输出系数得到初始化参数w b,其以标准正态分布随机变量的形式赋值"""
    w = torch.normal(mean,std,size=(input_num,output),requires_grad=True)
    b = torch.zeros(output,requires_grad=True)
    return w,b

def get_data_iter_of_train_and_test(true_w = torch.ones(200)*0.01,true_b = 0.05,n_trian=20,n_test=100,batch_size=5):
    """通过true_w和ture_b生成train and test 的iter"""
    train_dataarrays = make_data_by_parameters_for_linear(true_w,true_b,n_trian)
    test_dataarrays = make_data_by_parameters_for_linear(true_w,true_b,n_test)
    train_iter = load_data_iter_by_dataarrays_by_frame(train_dataarrays,batch_size)
    test_iter = load_data_iter_by_dataarrays_by_frame(test_dataarrays,batch_size)
    return train_iter,test_iter

def train_for_see_damping_by_frame(wd=0,lr=0.02,epochs=5,dim=200):
    """设置权重衰减系数,通过线性函数观察效果"""
    train_iter,test_iter = get_data_iter_of_train_and_test()
    net = nn.Sequential(nn.Linear(dim,1))
    def weight_init(m):
        if(type(m)==nn.Linear):
            m.weight.data.normal_(0,1)
    net.apply(weight_init)
    loss = nn.MSELoss(reduction='none')
    optimizer = torch.optim.SGD([{'params':net[0].weight,'weight_decay':wd},{'params':net[0].bias}],lr=lr)
    train_for_some_epoch_for_regression(epochs,net,loss,optimizer,train_iter,test_iter)
    print('w的L2范数为: ',net[0].weight.norm().item())
  

def penalty(w):
    """权重衰减自定义实现,L2范数"""
    return torch.sum(w.pow(2))/2
  
def train_for_see_damping_by_gxl(lambd=0,lr = 0.002,epochs = 5,dim=200):
    """设置权重衰减系数,通过线性函数观察效果,自定义实现"""
    w,b = get_init_params(dim,1)
    net = line_mode(w,b)
    loss = squared_loss
    optimizer = sgd((w,b),lr)
    animator = Animator()
    train_iter,test_iter = get_data_iter_of_train_and_test()
    """测试权重衰减"""
    for epoch in range(epochs):
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y) + lambd*penalty(w)
            l.mean().backward()
            optimizer()
        animator.add_dynamic(epoch+1,evaluate_loss_for_net(net,loss,train_iter),evaluate_loss_for_net(net,loss,test_iter))
    print(f'w的范数为:{torch.norm(w).item()}')


def dropout_layer(X,dropout):
    if(dropout==1):
        return torch.zeros_like(X)
    if(dropout==0):
        return X
    mask = (torch.rand(X.shape)>dropout).float()
    return (X * mask)/(1-dropout)


class MyDeepNet():
    """定义类时先输入输入输出参数,如果有隐藏层再以此输入各自隐藏层的输入,
    如果隐藏层想使用暂退法,则再以list的形式输入各个层的drop概率,dropsout参数必须显式声明,
    因为其前面是动态参数hiddens_input"""
    def __init__(self,input_num,output_num,*hiddens_input,dropouts=None,is_trian=True) -> None:
        self.training = is_trian
        self.dropouts = dropouts
        self.linList = []
        self.input_num = input_num
        self.output_num = output_num
        lens = len(hiddens_input)
        if lens==0:
            self.linList.append(nn.Linear(self.input_num,self.output_num))
        else:
            for i in range(lens):
                if(i==0):
                    self.linList.append(nn.Linear(self.input_num,hiddens_input[i]))
                else:
                    self.linList.append(nn.Linear(hiddens_input[i-1],hiddens_input[i]))
            self.linList.append(nn.Linear(hiddens_input[lens-1],self.output_num))
        self.relu = reLU_equation
    def __call__(self,X):
        Y = X.clone().reshape(-1,self.input_num)
        for i in range(len(self.linList)):
            if(i!= len(self.linList)-1):
                net = self.linList[i]
                Y = self.relu(net(Y))
                if(self.training==True):
                    if(self.dropouts==None):
                        Y = dropout_layer(Y,0.4)
                    else:
                        Y = dropout_layer(Y,self.dropouts[i])
            else:
                net = self.linList[i]
                Y = (net(Y))
        return Y
    def parameters(self):
        params = []
        for i in range(len(self.linList)):
            net = self.linList[i]
            params.append(net.weight)
            params.append(net.bias)
        return params


