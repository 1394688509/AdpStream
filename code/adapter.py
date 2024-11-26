import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.nonlinearity = nn.ReLU()

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        # 从均值为0，标准差为0.1的正态分布中初始化down_project的参数
        nn.init.normal_(self.down_project.weight, mean=0, std=0.1)
        # 如果down_project有bias，也初始化bias
        if self.down_project.bias is not None:
            nn.init.normal_(self.down_project.bias, mean=0, std=0.1)
        
        # 从均值为0，标准差为0.1的正态分布中初始化up_project的参数
        nn.init.normal_(self.up_project.weight, mean=0, std=0.1)
        # 如果up_project有bias，也初始化bias
        if self.up_project.bias is not None:
            nn.init.normal_(self.up_project.bias, mean=0, std=0.1)

    def forward(self, x):
        intermediate = self.nonlinearity(self.down_project(x))
        return self.up_project(intermediate)
    
