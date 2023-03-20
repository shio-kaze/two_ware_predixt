import torch
import numpy
import pandas
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import random_split, TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

class CNN_1D(nn.Module):
    def __init__(self):
        super(CNN_1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=32, kernel_size=1, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2, stride=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv6 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=1)
        self.pool6 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.fc = nn.Sequential(nn.Linear(in_features=224, out_features=3), nn.ReLU())
        # self.fc=nn.Sequential(nn.Linear(in_features=224,out_features=1))
        # self.net=nn.Sequential(self.conv1,nn.ReLU(),self.pool1,nn.ReLU(),self.conv2,nn.ReLU(),self.pool2,nn.ReLU(),self.conv3,nn.ReLU(),self.pool3,nn.ReLU(),
        #                        self.conv4,nn.ReLU(),self.pool4,nn.ReLU(),self.conv5,nn.ReLU(),self.pool5,nn.ReLU(),self.conv6,nn.ReLU(),self.pool6,nn.ReLU())
        self.net = nn.Sequential(self.conv1, nn.ReLU(), self.pool1, self.conv2, nn.ReLU(), self.pool2, self.conv3,
                                 nn.ReLU(), self.pool3,
                                 self.conv4, nn.ReLU(), self.pool4, self.conv5, nn.ReLU(), self.pool5, self.conv6,
                                 nn.ReLU(), self.pool6)

    def forward(self, x):
        x = self.net(x)
        out = self.fc(x.view(x.size(0), -1))
        return out


c1_feature = numpy.load('/kaggle/input/7-1dcnn/c1(31575000).npy')
c1_feature = torch.tensor(c1_feature, dtype=torch.float32)
c4_feature = numpy.load('/kaggle/input/7-1dcnn/c4(31575000).npy')
c4_feature = torch.tensor(c4_feature, dtype=torch.float32)
c6_feature = numpy.load('/kaggle/input/7-1dcnn/c6(31575000).npy')
c6_feature = torch.tensor(c6_feature, dtype=torch.float32)
c1_feature, c4_feature, c6_feature = c1_feature.cuda(), c4_feature.cuda(), c6_feature.cuda()
feature = torch.concat([c1_feature, c4_feature, c6_feature], dim=0)

c1_actual = pandas.read_csv('/kaggle/input/labels/c1_wear.csv', index_col=0, header=0,
                            names=['f1', 'f2', 'f3'])
c1_actual = c1_actual.values.reshape(-1, 3)

c4_actual = pandas.read_csv('/kaggle/input/labels/c4_wear.csv', index_col=0, header=0,
                            names=['f1', 'f2', 'f3'])
c4_actual = c4_actual.values.reshape(-1, 3)

c6_actual = pandas.read_csv('/kaggle/input/labels/c6_wear.csv', index_col=0, header=0,
                            names=['f1', 'f2', 'f3'])
c6_actual = c6_actual.values.reshape(-1, 3)
wear_ndarray=numpy.concatenate([c1_actual,c4_actual,c6_actual],axis=0)
scaler=MinMaxScaler()
wear_ndarray=scaler.fit_transform(wear_ndarray)
wear = torch.tensor(wear_ndarray,dtype=torch.float32)
wear=wear.cuda()

# feature=feature.cuda()

batch_size = 10
epoch = 1500
lr = 0.001
dataset = TensorDataset(feature, wear)
train_num = int(len(wear) * 0.8)
train_set, val_set = random_split(dataset, [train_num, len(wear) - train_num])
train_loader = DataLoader(train_set, batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size, shuffle=True)

model = CNN_1D()
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
# scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
loss_func = nn.MSELoss()

train_loss = []

for i in range(epoch):
    model.train()
    loss_train = 0
    for train_x, train_y in train_loader:
        out_train = model(train_x)
        loss = loss_func(out_train, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         scheduler.step()
        loss_train += loss.item()
    loss_train /= len(train_loader)
    train_loss.append(loss_train)

    loss_val = 0
    with torch.no_grad():
        model.eval()
        for val_x, val_y in val_loader:
            out_val = model(val_x)
            loss_ = loss_func(out_val, val_y)
            loss_val += loss_.item()
        loss_val /= len(val_loader)
    print('epoch %s train_loss:%.6f val_loss:%6f' % (i, loss_train, loss_val))

# 损失曲线
plt.figure(figsize=(8, 4))
plt.plot(train_loss)
plt.title('train')
plt.savefig('/kaggle/working/loss.png')

# 预测
with torch.no_grad():
    c1_pred = model(c1_feature).detach().cpu().numpy()
    c4_pred = model(c4_feature).detach().cpu().numpy()
    c6_pred = model(c6_feature).detach().cpu().numpy()
c1_pred=scaler.inverse_transform(c1_pred)
c4_pred=scaler.inverse_transform(c4_pred)
c6_pred=scaler.inverse_transform(c6_pred)
rmse1 = numpy.sqrt(mean_squared_error(c1_pred, c1_actual))
rmse2 = numpy.sqrt(mean_squared_error(c4_pred, c4_actual))
rmse3 = numpy.sqrt(mean_squared_error(c6_pred, c6_actual))

x = numpy.arange(315)
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(x, c1_actual, label='actual')
plt.plot(x, c1_pred, label='pred')
plt.legend()
plt.title('c1')

plt.subplot(1, 3, 2)
plt.plot(x, c4_actual, label='actual')
plt.plot(x, c4_pred, label='pred')
plt.legend()
plt.title('c4')

plt.subplot(1, 3, 3)
plt.plot(x, c6_actual, label='actual')
plt.plot(x, c6_pred, label='pred')
plt.legend()
plt.title('c6')

plt.savefig('/kaggle/working/result.png')
print('c1 rms :%s' % rmse1)
print('c4 rms :%s' % rmse2)
print('c6 rms :%s' % rmse3)
