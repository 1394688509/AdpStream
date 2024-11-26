import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn import metrics
import scipy.spatial as sp
from torch.autograd import Variable
import argparse
import scipy.io
from scipy.special import ndtr
import csv
from Loss_window import WindowLoss
from adapter import Adapter

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='NSL')
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument("--dev", help="device", default="cuda:0")
    parser.add_argument("--epochs", type=int, help="number of epochs for ae", default=8000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--memlen", type=int, help="size of memory", default=2048)
    parser.add_argument("--win_size", type=int, help="size of window", default=50) # 记录loss
    parser.add_argument("--skip_threshold", type=float, help="threshold", default=1)
    parser.add_argument("--dim", type=int, help="dimension of encoder_output", default=8)
    parser.add_argument("--b_dim", type=int, help="dimension of adapter_embedding", default=32)
    parser.add_argument("--seed", type=int, help="random seed", default=0)
    parser.add_argument("--gamma", type=float, help="knn coefficient", default=0.5)
    args = parser.parse_args()
    
    return args

args = arg_parse()

# torch.manual_seed(args.seed)
nfile = None
lfile = None
if args.dataset == 'NSL':
    # nfile = 'data/nsl.txt'
    # lfile = 'data/nsllabel.txt'
    nfile = '../data/nsl.txt'
    lfile = '../data/nsllabel.txt'
elif args.dataset == 'KDD':
    nfile = '../data/kdd.txt'
    lfile = '../data/kddlabel.txt'
elif args.dataset == 'UNSW':
    nfile = '../data/unsw.txt'
    lfile = '../data/unswlabel.txt'
elif args.dataset == 'DOS':
    nfile = '../data/dos.txt'
    lfile = '../data/doslabel.txt'
else:
    df = scipy.io.loadmat('../data/'+args.dataset+".mat")
    # df = scipy.io.loadmat('data/'+args.dataset+".mat")
    numeric = torch.FloatTensor(df['X'])
    labels = (df['y']).astype(float).reshape(-1)

device = torch.device(args.dev)

class MemStream(nn.Module):
    def __init__(self, in_dim, params):
        super(MemStream, self).__init__()
        self.params = params
        self.dim = params['dim']
        self.in_dim = in_dim
        self.bottleneck_dim = params['b_dim']
        self.out_dim = int(in_dim//self.dim)
        self.memory_len = params['memory_len']
        self.win_size = params['win_size']
        self.skip_threshold = params['skip_threshold']
        self.gamma = params['gamma']
        # self.max_thres = torch.tensor(params['beta']).to(device)
        self.memory = torch.randn(self.memory_len, self.out_dim).to(device)
        self.mem_data = torch.randn(self.memory_len, self.in_dim).to(device)
        self.memory.requires_grad = False
        self.mem_data.requires_grad = False
        self.batch_size = params['memory_len']
        self.num_mem_update = 0
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.Tanh(),
        ).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(self.out_dim, self.in_dim)
        ).to(device)
        self.clock = 0
        self.last_update = -1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['lr'])
        self.loss_fn = nn.MSELoss()
        self.win_loss = WindowLoss(self.win_size)
        # self.count = 0
        self.K = 3
        self.exp = torch.Tensor([self.gamma**i for i in range(self.K)]).to(device)


    def train_autoencoder(self, data, epochs):
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (data - self.mean) / self.std
        new[:, self.std == 0] = 0
        new = Variable(new)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.decoder(self.encoder(new + 0.001*torch.randn_like(new).to(device)))
            loss = self.loss_fn(output, new)
            loss.backward()
            self.optimizer.step()


    def update_memory(self, output_loss, rec_loss, encoder_output, data, index):
        if output_loss <= rec_loss:
            """ FIFO ——> 替换最相近的点:topk()根据index进行替换  """
            # least_used_pos = self.count%self.memory_len
            least_used_pos = index
            self.memory[least_used_pos] = encoder_output
            self.mem_data[least_used_pos] = data
            self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
            self.count += 1
            return 1
        return 0

    def initialize_memory(self, x):
        mean, std = model.mem_data.mean(0), model.mem_data.std(0)
        new = (x - mean) / std
        new[:, std == 0] = 0
        self.memory = self.encoder(new)
        # self.memory.requires_grad = False
        self.mem_data = x

    def window_loss(self, x_emb, x):
        """ 捕获短时依赖 """
        rec_x = self.decoder(x_emb)
        loss = self.loss_fn(rec_x, x)
        loss_item = loss.item()
        if self.win_loss is not None:
            mean = self.win_loss.mean
            std = (
                self.win_loss.sample_std if self.win_loss.population_std > 0 else 1
            )
            self.win_loss.update(loss_item)

        loss_scaled = (loss_item - mean) / std
        prob = ndtr(loss_scaled)
        win_loss = (self.skip_threshold - prob) / self.skip_threshold * loss
        return win_loss
    
    def forward(self, x):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        encoder_output = self.encoder(new)
        
        loss = self.window_loss(encoder_output, new)
        
        distances, index = torch.topk(torch.norm(self.memory - encoder_output, dim=1, p=1), k=self.K, largest=False)
        # print(index[0])
        
#         loss_values = torch.norm(self.memory - encoder_output, dim=1, p=1).min()
        # loss_values = (torch.topk(torch.norm(self.memory - encoder_output, dim=1, p=1), k=self.K, largest=False)[0]*self.exp).sum()/self.exp.sum()
        
        loss_values = (distances*self.exp).sum()/self.exp.sum()
        
        self.update_memory(loss_values, loss, encoder_output, x, index[0])
        
        loss_combined = loss + loss_values
        
        return loss_combined
    
    
class AdpStream(MemStream):
    def __init__(self, in_dim, params):
        super(AdpStream, self).__init__(in_dim, params)
        # 添加 Adapter 实例
        self.adapter = Adapter(self.out_dim, self.bottleneck_dim)
        # 将 Adapter 的参数设为不需要梯度
        for param in self.adapter.parameters():
            # param.requires_grad = True
            param.requires_grad = False
    
    def window_loss(self, x_emb, x):
        """ 捕获短时依赖 """
        rec_x = self.decoder(x_emb)
        loss = self.loss_fn(rec_x, x)
        loss_item = loss.item()
        if self.win_loss is not None:
            mean = self.win_loss.mean
            std = (
                self.win_loss.sample_std if self.win_loss.population_std > 0 else 1
            )
            self.win_loss.update(loss_item)

        loss_scaled = (loss_item - mean) / std
        prob = ndtr(loss_scaled)
        win_loss = (self.skip_threshold - prob) / self.skip_threshold * loss
        return win_loss, loss

    def update_memory_with_adapter(self, output_loss, rec_loss, adapted_output, data, index):
        # 使用 adapted_output 替代 encoder_output 来更新记忆
        if output_loss <= rec_loss:
            """ FIFO ——> 替换最相近的点:topk()根据index进行替换  """
            # least_used_pos = self.count%self.memory_len
            least_used_pos = index
            self.memory[least_used_pos] = adapted_output
            self.mem_data[least_used_pos] = data
            self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
            # self.count += 1
            return 1
        # else:
        #     for param in model.adapter.parameters():
        #         param.requires_grad = True
        return 0
    
    def forward(self, x):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        encoder_output = self.encoder(new)

        # 使用 Adapter 更新编码器的输出
        adapted_output = self.adapter(encoder_output)

        loss,model_loss = self.window_loss(adapted_output, new)

        distances, index = torch.topk(torch.norm(self.memory - adapted_output, dim=1, p=1), k=self.K, largest=False)
        # print(index[0])

#         loss_values = torch.norm(self.memory - encoder_output, dim=1, p=1).min()
        # loss_values = (torch.topk(torch.norm(self.memory - encoder_output, dim=1, p=1), k=self.K, largest=False)[0]*self.exp).sum()/self.exp.sum()

        loss_values = (distances*self.exp).sum()/self.exp.sum()

        self.update_memory_with_adapter(loss_values, loss, adapted_output, x, index[0])

        loss_combined = loss + loss_values
        
        return loss_combined,model_loss



if args.dataset in ['KDD', 'NSL', 'UNSW', 'DOS', 'SYN']:
    numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ','))
    labels = np.loadtxt(lfile, delimiter=',')

if args.dataset == 'KDD':
    labels = 1 - labels

torch.manual_seed(args.seed)
N = args.memlen
params = {
        'beta': args.beta, 'memory_len': args.memlen, 'win_size':args.win_size, 'skip_threshold':args.skip_threshold, 
        'batch_size':1, 'lr':args.lr, 'gamma':args.gamma, 'dim':args.dim, 'b_dim':args.b_dim,
        }

model = AdpStream(numeric[0].shape[0],params).to(device)

# def model_structure(model):
#     blank = ' '
#     print('-' * 133)
#     print('|' + ' ' * 20 + 'weight name' + ' ' * 20 + '|' \
#           + ' ' * 20 + 'weight shape' + ' ' * 20 + '|' \
#           + ' ' * 10 + 'number' + ' ' * 10 + '|')
#     print('-' * 133)
#     num_para = 0
#     type_size = 1

#     for index, (key, w_variable) in enumerate(model.named_parameters()):
#         if len(key) <= 30:
#             key = key + (30 - len(key)) * blank
#         shape = str(w_variable.shape)
#         if len(shape) <= 40:
#             shape = shape + (40 - len(shape)) * blank
#         each_para = 1
#         for k in w_variable.shape:
#             each_para *= k
#         num_para += each_para
#         str_num = str(each_para)
#         if len(str_num) <= 10:
#             str_num = str_num + (10 - len(str_num)) * blank

#         print('| {}\t\t\t\t| {}\t\t| {}\t\t|'.format(key, shape, str_num))
#     print('-' * 133)
#     print('The total number of parameters: ' + str(num_para))
#     print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
#     print('-' * 133)


# model_structure(model)
# # 提前退出程序
# exit(0)


batch_size = params['batch_size']
# print(args.dataset, lr, memlen, win_size, gamma)
print(args.dataset, args.lr, args.memlen, args.win_size, args.gamma, args.dim, args.b_dim)
data_loader = DataLoader(numeric, batch_size=batch_size)
init_data = numeric[labels == 0][:N].to(device)
model.mem_data = init_data
# 预训练 AE
model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs)
# 预训练完成后，初始化记忆
model.initialize_memory(Variable(init_data[:N]))
# 冻结除了 Adapter 之外的所有参数
for param in model.parameters():
    if param.requires_grad:
        param.requires_grad = False
# 只对 Adapter 层的参数开启梯度计算
for param in model.adapter.parameters():
    param.requires_grad = True
err = []

my_optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

for data in data_loader:
    my_optimizer.zero_grad()
    output,model_loss = model(data.to(device))
    model_loss.backward(retain_graph=True)
    my_optimizer.step()
    err.append(output.detach().cpu().numpy())                      
scores = np.array([i for i in err])
auc = metrics.roc_auc_score(labels, scores)
print("ROC-AUC", auc)


