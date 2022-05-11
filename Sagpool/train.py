import torch
from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from model import Net
import torch.nn.functional as F

def evaluate(args,model,loader):
    model.eval()
    correct=0
    loss=0
    for data in loader:
        data=data.to(args.device)
        out=model(data)
        pred=out.max(dim=1)[1]
        correct+=pred.eq(data.y).sum().item()
        loss+=F.nll_loss(out,data.y,reduction='sum').item()
    return correct/len(loader.dataset),loss/len(loader.dataset)

def train(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset=TUDataset('datasets/',name=args.dataset)
    args.num_classes=dataset.num_classes
    args.num_feats=dataset.num_features

    num_training=int(len(dataset)*0.8)
    num_val=int(len(dataset)*0.1)
    num_eval=len(dataset)-num_training-num_val
    # 划分训练，验证，测试集
    training_set,validation_set,evaluation_set=random_split(dataset,[num_training,num_val,num_eval])

    train_loader=DataLoader(training_set,batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(evaluation_set, batch_size=1, shuffle=True)

    model=Net(args).to(device=args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    min_loss = 1e10
    patience = 0

    for epoch in range(args.epochs):
        model.train()
        #按加载器加载
        for i,data in enumerate(train_loader):
            data=data.to(args.device)
            out=model(data)
            loss=F.nll_loss(out,data.y)
            print('epoch:',epoch,'batch:',i,'loss:',loss.data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        val_acc,val_loss=evaluate(args,model,val_loader)
        print('val loss:',val_loss,' val accuracy:',val_acc)

        if val_loss<min_loss:
            torch.save(model.state_dict(),'latest_great.pth')
            print("model saved at epoch_",epoch)
            min_loss=val_loss
            patience=0
        else:
            patience+=1
        if patience>args.patience:
            break

    test_model(args,model,eval_loader)

def test_model(args,model,testloader):
    test_acc,test_loss=evaluate(args,model,testloader)
    print('test accuracy:',test_acc,'test loss:',test_loss)