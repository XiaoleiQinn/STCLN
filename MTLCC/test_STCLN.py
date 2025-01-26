import sys
sys.path.append("/data/xiaolei.qin/Projects/mtlcc/MTLCC-pytorch-master/src")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from utils.dataset import ijgiDataset as Dataset
from STCLN import UTAE,UTAEClassification
import numpy as np
from logger import Logger, Printer, VisdomLogger
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.deterministic = True
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, cohen_kappa_score,f1_score
import pandas as pd

snapshot1 = torch.load(
    '/data/xiaolei.qin/Projects/mtlcc/checkpoints/Mtlcc/SSpv/finetune_wptimespace0.4/2Per_48Is/STCLN/model_best.tar')

model_state1 = snapshot1.pop('model_state_dict', snapshot1)

network1 = UTAEClassification(UTAE(n_channels=13, n_classes=18, bilinear=True,
                 encoder_widths=[32, 256],
                 decoder_widths=[32, 256],
                 agg_mode="att_group",
                 n_head=8,
                 d_model=256,
                 d_k=32))

network1.load_state_dict(model_state1)

if torch.cuda.is_available():

    network1 = network1.cuda()

network1.eval()
dataset = Dataset("/data/xiaolei.qin/Projects/mtlcc/MTLCC-pytorch-master/src/data",
                  tileids="tileids/eval.tileids")


# --------------------------------------------------
dataloader = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=False,num_workers=4)

printer = Printer(N=len(dataloader))

iterator = iter(dataloader)
firstIter = True
Loss_list =0
with torch.no_grad():
    for iteration, data in enumerate(dataloader):

        input, target = data

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        #
        output,*_=network1.forward(input, torch.tensor(range(input.shape[1])).unsqueeze(0).repeat(input.shape[0], 1).cuda())#SLTN

        output = output.argmax(dim=1)

        if firstIter:
            y_true = target.cpu().detach().numpy()
            y_pred = output.cpu().detach().numpy()
            firstIter = False
        else:
            y_true = np.concatenate((target.cpu().detach().numpy(), y_true), 0)
            y_pred = np.concatenate((output.cpu().detach().numpy(), y_pred), 0)
        printer.print(None, iteration)

y_true=y_true.flatten()
y_pred=y_pred.flatten()
y_true0=y_true.copy()
y_pred0=y_pred.copy()
y_true=y_true[(y_true0>0)&(y_pred0>0)]
y_pred = y_pred[(y_true0>0)&(y_pred0>0)]
labels = torch.from_numpy(y_true).unique()

C=confusion_matrix(y_true, y_pred,labels=labels)


Precision = precision_score(y_true, y_pred,average=None)
Recall = recall_score(y_true, y_pred,average=None)
f1_scores = f1_score(y_true, y_pred,average=None)
meanf1_score = f1_score(y_true, y_pred,average="macro")

Acc = accuracy_score(y_true,y_pred)

print('Acc','%.4f'%Acc)
conf_matrix = C
true_positive = np.diag(conf_matrix)
false_positive = np.sum(conf_matrix, 0) - true_positive
false_negative = np.sum(conf_matrix, 1) - true_positive

# Just in case we get a division by 0, ignore/hide the error
with np.errstate(divide='ignore', invalid='ignore'):
    iou = true_positive / (true_positive + false_positive + false_negative)
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    f1_score = 2 * recall * precision / (recall + precision)
miou = float(np.nanmean(iou))
mf_score = float(np.nanmean(f1_score))
print('iou', [x for x in iou])
print('miou','%.4f'%miou)
print('f1_score', f1_score)
print('meanf1_score','%.4f'%meanf1_score)

outacc=pd.DataFrame({'UA': Precision,'PA':Recall,'IOU':iou,'F1-score':f1_scores})


