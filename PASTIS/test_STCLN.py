import sys
sys.path.append('/data/xiaolei.qin/Projects/mtlcc/utae-paps-main')
from src.dataset import PASTIS_Dataset
from src import utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
from STCLN import UTAE,UTAEClassification
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, cohen_kappa_score,f1_score
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.deterministic = True


snapshot1 = torch.load(
    '/data/xiaolei.qin/Projects/mtlcc/checkpoints/Pastis/SSpv/finetune_wptimespace/2Per_32Is/STLN/model_best.tar')

model_state1 = snapshot1.pop('model_state_dict', snapshot1)

network1 = UTAEClassification(UTAE(n_channels=10, n_classes=20, bilinear=True,
                 encoder_widths=[32, 256],
                 decoder_widths=[32, 256],
                 agg_mode="att_group",
                 n_head=8,
                 d_model=256,
                 d_k=64))


network1.load_state_dict(model_state1)

loss = torch.nn.NLLLoss()
if torch.cuda.is_available():
    network1 = torch.nn.DataParallel(network1,device_ids=[0]).cuda()

network1.eval()

# #pastis---------------------------
collate_fn = lambda x: utils.pad_collate(x, pad_value=0)
dataset = PASTIS_Dataset(folder='/data/xiaolei.qin/Projects/CropClassification/Data/PASTIS/Data/PASTIS', norm=True,reference_date='2018-09-01',
                              target='semantic', folds=[4])


dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2, collate_fn=collate_fn)
iterator = iter(dataloader)
firstIter = True
Loss_list =0
with torch.no_grad():
    for m in range(0,len(dataloader)):
        print('m',m,len(dataloader))
        (data_, dates_), label_ = next(iterator)
        for i in range(1):
            for j in range(1):

                split = data_.shape[-1] // 1
                data = data_[:, :, :, i * split:(i + 1) * split, j * split:(j + 1) * split]
                label = label_[:,i * split:(i + 1) * split, j * split:(j + 1) * split]
                if torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()


                output,_ = network1.forward(data, torch.tensor(range(data.shape[1])).unsqueeze(0).repeat(data.shape[0], 1).cuda())
                output = output.argmax(dim=1)

            #
                if firstIter:
                    y_true = label.cpu().detach().numpy()
                    y_pred = output.cpu().detach().numpy()
                    firstIter = False
                else:
                    y_true = np.concatenate((label.cpu().detach().numpy(), y_true), 0)
                    y_pred = np.concatenate((output.cpu().detach().numpy(), y_pred), 0)

y_true=y_true.flatten()
y_pred=y_pred.flatten()
y_true0=y_true.copy()
y_pred0=y_pred.copy()
y_true=y_true[(y_true0!=19)&(y_true0!=0)]
y_pred = y_pred[(y_true0!=19)&(y_true0!=0)]
labels = torch.from_numpy(y_true).unique()

labelslist = []
for l in labels:
    labelslist.append(str(l.cpu().numpy()))

his=np.histogram(y_true,bins=range(0,21),range=None,weights=None,density=False)
print(his)

C=confusion_matrix(y_true, y_pred,labels=labels)

Precision = precision_score(y_true, y_pred,average=None)
Recall = recall_score(y_true, y_pred,average=None)
f1_scores = f1_score(y_true, y_pred,average=None)
meanf1_score = f1_score(y_true, y_pred,average="macro")

Acc = accuracy_score(y_true,y_pred)

print(Acc,'%.4f'%Acc)

Kappa = cohen_kappa_score(y_true,y_pred)

print(Kappa)

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
mf_score=float(np.nanmean(f1_score))
print('iou',[x for x in iou])
print(miou,'%.4f'%miou)
print('f1_score',f1_score)
print(meanf1_score,'%.4f'%meanf1_score)


