import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from skmultiflow.data.data_stream import DataStream
from skmultiflow.trees.hoeffding_tree import HoeffdingTreeClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import MinMaxScaler

def drift_detector(S,T,threshold=0.75):
    """
    :param S: 旧数据
    :param T: 新数据
    :param threshold: AUC的阈值
    :return: 是否发生了概念漂移
    """

    T=pd.DataFrame(T)
    S=pd.DataFrame(S)

    # 分别给新旧数据打标，旧的打0，新的打1
    S['in_target']=0
    T['in_target']=1

    # 将两个数据拼接到一起去
    TS=pd.concat([T,S],axis=0,ignore_index=True)

    labels=TS['in_target'].values
    TS=TS.drop('in_target',axis=1).values

    clf=LogisticRegression(solver='liblinear')
    # 记录测试集的预测结果
    predictions=np.zeros((int(labels.shape[0]),))

    # 将TS分成两个相同大小的块
    skf=StratifiedKFold(n_splits=2,shuffle=True)
    # 获取训练集对应的索引和测试集对应的索引
    for train_idx ,test_idx in skf.split(TS,labels):
        X_train,X_test=TS[train_idx],TS[test_idx]
        y_train,y_test=labels[train_idx],labels[test_idx]
        clf.fit(X_train,y_train)
        # 只获取预测出来的为1的概率
        probs=clf.predict_proba(X_test)[:,1]

        predictions[test_idx]=probs

    # auc_score = AUC(labels, predictions)
    # 这里原代码把没有预测的部分也拿出来计算auc了？
    # 我将其中属于测试集的部分拿出来求AUC了
    auc_score=AUC(labels[np.nonzero(predictions)[0]],predictions[np.nonzero(predictions)[0]])

    if auc_score>threshold:
        return True
    else:
        return False

class D3():
    def __init__(self,w,rho,dim,auc):
        """
        :param w: 旧数据的数量
        :param rho: 新数据占旧数据的比例
        :param dim: 数据的特征数量
        :param auc:
        """
        self.size=int(w*(1+rho))
        self.win_data=np.zeros((self.size,dim))
        self.win_label=np.zeros(self.size)
        self.w=w
        self.rho=rho
        self.dim=dim
        self.auc=auc
        self.drift_count=0
        self.window_index=0

    def addInstance(self,X,y):
        if self.isEmpty():
            self.win_data[self.window_index]=X
            self.win_label[self.window_index]=y
            self.window_index=self.window_index+1

    def isEmpty(self):
        return self.window_index<self.size

    def driftCheck(self):
        # 该函数中的丢弃数据实际上只是逻辑上丢弃，通过控制window_index来实现逻辑上丢弃数据
        if drift_detector(self.win_data[:self.w],self.win_data[self.w:],self.auc):
            self.window_index=int(self.w*self.rho)
            self.win_data=np.roll(self.win_data,-1*self.w,axis=0)
            self.win_label=np.roll(self.win_label,-1*self.w,axis=0)
            self.drift_count+=1
            return True
        else:
            self.window_index=self.w
            self.win_data=np.roll(self.win_data,-1*int(self.w*self.rho),axis=0)
            self.win_label=np.roll(self.win_label,-1*int(self.w*self.rho),axis=0)
            return False

    def getCurrentData(self):
        return self.win_data[:self.window_index]

    def getCurrentLabels(self):
        return self.win_label[:self.window_index]

# 选择使用哪个数据库
def select_data(x):
    df=pd.read_csv(x)
    scaler=MinMaxScaler()
    # 最后一列是标签 不需要
    df.iloc[:,0:df.shape[1]-1]=scaler.fit_transform(df.iloc[:,0:df.shape[1]-1])
    return df

def check_true(y,y_hat):
    if y==y_hat:
        return 1
    else :
        return 0

# 将数据分成一组N个，计算这N个的平均值
def window_average(x,N):
    low_index=0
    high_index=low_index+N
    w_avg=[]
    while(high_index<len(x)):
        temp=sum(x[low_index:high_index])/N
        w_avg.append(temp)
        low_index+=N
        high_index+=N
    return w_avg

if __name__=="__main__":
    argpaser=argparse.ArgumentParser("D3")
    argpaser.add_argument('--filename',type=str,default='./artificial/movingRBF.csv',help='choose which dataset you want to use')
    argpaser.add_argument('--old_num', type=int, default=100, help='choose how much old sample you want to use')
    argpaser.add_argument('--ratio', type=float, default=0.3, help='choose the ratio of number of new samples and of samples')
    argpaser.add_argument('--auc',type=float,default=0.75,help='choose the auc threshold')
    args=argpaser.parse_args()

    df=select_data(args.filename)
    # 准备好流式数据
    stream=DataStream(df)
    stream.prepare_for_use()
    stream_clf=HoeffdingTreeClassifier()
    w=args.old_num
    rho=args.ratio
    auc=args.auc

    D3_win=D3(w,rho,stream.n_features,auc)
    stream_acc=[]
    stream_record=[]
    stream_true=0

    i=0
    start=time.time()
    writer = SummaryWriter()
    X,y=stream.next_sample(int(w*rho))
    stream_clf.partial_fit(X,y,classes=stream.target_values)


    while(stream.has_more_samples()):
        X,y=stream.next_sample()
        # 如果此时窗口未满，向其中冲入数据并且线上训练流分类器
        if D3_win.isEmpty():
            D3_win.addInstance(X,y)
            y_hat=stream_clf.predict(X)

            # 计算预测准确的数量
            stream_true+=check_true(y,y_hat)
            stream_clf.partial_fit(X,y)
            # 计算预测的准确率
            stream_acc.append(stream_true/(i+1))
            stream_record.append(check_true(y,y_hat))

            writer.add_scalars('',{'stream_acc':stream_acc[-1],
                                                   'stream_record':stream_true},i)

        # 如果窗口已经满了，仍然要做流式训练，但同时也需要判断是否漂移
        else:
            y_hat=stream_clf.predict(X)
            stream_true+=check_true(y,y_hat)
            stream_clf.partial_fit(X,y)
            stream_acc.append(stream_true/(i+1))
            stream_record.append(check_true(y,y_hat))
            D3_win.driftCheck()

            writer.add_scalars('', {'stream_acc': stream_acc[-1],
                                                    'stream_record': stream_true,
                                                    'drift_count':D3_win.drift_count},i)

            D3_win.addInstance(X,y)
        i+=1

    elapsed = format(time.time() - start, '.4f')
    acc = format((stream_acc[-1] * 100), '.4f')
    print(f"Final accuracy:{acc},Elasped time:{elapsed}")
    writer.close()





