import torch
from torch import nn

class AttentionalAggregation(nn.Module):
    def __init__(self, dim, out_ele_num=1):
        super().__init__()

        self.out_ele_num = out_ele_num
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(in_features=dim, out_features=out_ele_num * dim)

    def forward(self, x):
        in_ele_num = x.shape[1]
        in_ele_len = int(x.shape[2])
        out_ele_len = in_ele_len

        x_1st = x
        x_1st_tp = x_1st.reshape(-1, in_ele_len)
        weights_1st = self.fc(x_1st_tp)

        weights_1st = weights_1st.reshape(-1, in_ele_num, self.out_ele_num, out_ele_len)
        weights_1st = self.softmax(weights_1st)
        x_1st = x_1st[:, :, None, :].repeat(1, 1, self.out_ele_num, 1)
        x_1st = x_1st * weights_1st
        x_1st = torch.sum(x_1st, dim=1)
        x_1st.reshape(-1, self.out_ele_num * out_ele_len)

        return torch.squeeze(x_1st)
