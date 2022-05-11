import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from layers import SAGPool
import torch_geometric
from torch import nn
import torch
from torch_geometric.nn import global_mean_pool, global_max_pool

class Net(nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.args=args
        self.num_feats=args.num_feats
        self.nhid=args.num_hid
        self.num_classes=args.num_classes
        self.pool_ratio=args.pooling_ratio
        self.dropout=args.dropout

        self.conv1=GCNConv(self.num_feats,self.nhid)
        self.pool1=SAGPool(self.nhid,ratio=self.pool_ratio)
        self.conv2=GCNConv(self.nhid,self.nhid)
        self.pool2=SAGPool(self.nhid,ratio=self.pool_ratio)
        self.conv3=GCNConv(self.nhid,self.nhid)
        self.pool3=SAGPool(self.nhid,ratio=self.pool_ratio)

        self.lin1=nn.Linear(self.nhid*2,self.nhid)
        self.lin2=nn.Linear(self.nhid,self.nhid//2)
        self.lin3=nn.Linear(self.nhid//2,self.num_classes)

    def forward(self,data:torch_geometric.data.Data):
        x,edge_index,batch=data.x,data.edge_index,data.batch

        x=F.relu(self.conv1(x,edge_index))
        x,edge_index,_,batch,_=self.pool1.forward(x,edge_index,None,batch)
        # readout
        x1=torch.cat([global_max_pool(x,batch),global_mean_pool(x,batch)],dim=1)

        x=F.relu(self.conv2(x,edge_index))
        x,edge_index,_,batch,_=self.pool2.forward(x,edge_index,None,batch)
        # readout
        x2=torch.cat([global_max_pool(x,batch),global_mean_pool(x,batch)],dim=1)

        x=F.relu(self.conv3(x,edge_index))
        x,edge_index,_,batch,_=self.pool3.forward(x,edge_index,None,batch)
        # readout
        x3=torch.cat([global_max_pool(x,batch),global_mean_pool(x,batch)],dim=1)

        x=x1+x2+x3

        x=F.relu(self.lin1(x))
        x=F.dropout(x,self.dropout,training=self.training)
        x=F.relu(self.lin2(x))
        x=F.log_softmax(F.relu(self.lin3(x)),dim=-1)

        return x