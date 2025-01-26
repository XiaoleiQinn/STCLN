import sys
sys.path.append('/data/xiaolei.qin/Projects/mtlcc/MTLCC-pytorch-master/src')
sys.path.append('/data/xiaolei.qin/Projects/mtlcc')

import numpy as np
import torch.nn
from torch.cuda.amp import autocast as autocast,GradScaler
from utils.dataset import ijgiDataset as Dataset
from STCLN import UTAE,UTAEClassification

from logger import Logger, Printer, VisdomLogger
import argparse

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
deviceID=[0]
torch.manual_seed(3407)
torch.backends.cudnn.deterministic = True
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("data", type=str,help="path to dataset")
    parser.add_argument('-data', "--datadir", default = "/data/xiaolei.qin/Projects/mtlcc/MTLCC-pytorch-master/src/data",type=str, help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=2, type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=1, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=100, type=int, help="epochs to train" )
    parser.add_argument('-l', "--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument('-p', "--pretrain_pth", default="/data/xiaolei.qin/Projects/mtlcc/checkpoints/Mtlcc/SSpv/pretrain/STCLN_largeIs/batchtimespace0.4/checkpoint_99.utae.tar", type=str, help="pretrain path")
    # /data/xiaolei.qin/Projects/mtlcc/checkpoints/Mtlcc/SSpv/pretrain/STCLN_largeIs/batchtimespace0.4/checkpoint_99.utae.tar #0.2,0.4,0.6,0.8
    # /data/xiaolei.qin/Projects/mtlcc/checkpoints/Mtlcc/SSpv/pretrain/STCLN_largeIs/batchtime0.4/checkpoint_99.utae.tar
    parser.add_argument('-s', "--snapshot", default=None, type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default="/data/xiaolei.qin/Projects/mtlcc/checkpoints/Mtlcc/SSpv/finetune_wptimespace0.4/2Per_48Is/STCLN", type=str, help="directory to save checkpoints")
    parser.add_argument('-runp', "--run_path", default="/data/xiaolei.qin/Projects/mtlcc/runs/Mtlcc/SSPV/finetune_wptimespace0.15/2Per_48Is/", type=str, help="run path")  #kmeans_lossClear
    return parser.parse_args()

def main(
    datadir,
    batchsize = 16,
    workers = 8,
    epochs = 100,
    lr = 1e-3,
    pretrain_pth = None,
    snapshot = None,
    checkpoint_dir = None,
    run_path=None
    ):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    #---------------mtlcc dataset-----------------------------
    # patch with largest distribution of crop type
    Train_type1=[1175,8813,4325,6045,11503,7935,537,9423,7382,7849,14417,7315,2672,16422,3134]
    Train_type2=[17003,13009,14073,14263,12921,13095,11037,12763,15142,10748,11597,16827,7984,12922,14794]
    Train_type4=[1934,13112,1933,9336,4442,14860,13029,7929,10113,10659,10258,17815,15453,13119,13743]
    Train_type5=[11473,11563,12164,11474,8175,10609,14527,8089,13657,11044,14528,10611,10248,11387,11476]
    Train_type6=[1675,1762,7275,7382,3384,6428,17643,2431,9529,5325,10327,4645,1943,7532,10321]
    Train_type7=[13738,11582,13396,9253,9430,8921,108,6422,14809,11423,4437,3241,5983,5986,6509]
    Train_type8=[14327,17952,5243,15722,2929,14892,11523,13007,12622,14035,15809,7097,14793,17584,17174]
    Train_type9=[15215,2359,10905,5334,5421,11436,5195,9299,11320,10202,11289,7065,4759,9386,7969]
    Train_type10=[4686,10542,13750,6173,6512,12880,5303,5194,5982,5809,3287,6085,3636,4062,4673]
    Train_type11=[10028,6857,14969,14883,6856,11349,11337,8893,11592,5757,536,11348,6800,14882,14944]
    Train_type12 = [3716,3802,3803,3894,3628,6172,2694,2488,4061,4325,6087,6870,3889,6694,6609]
    Train_type15=[4511,6600,7560,8088,6426,5390,10895,8457,5391,5304,5998,7537,10802,13323,5394]
    Train_type16 = [7244,532,13054,11507,12618,4705,8018,7065,14161,7755,11589,7502,7927,11679,6104]

    Train_collection = [Train_type1,Train_type2,Train_type4,Train_type5,Train_type6,Train_type7,Train_type8,Train_type9,Train_type10,
                        Train_type11,Train_type12,Train_type15,Train_type16]

    Val_type1 = [1007,14407,1976,920,929,6793,930,7777,1279,3316,5692,4015,10207,1744,1537]
    Val_type2=[9575,11954,3839,4016,11629,16572,7888,9574,12265,17202,16575,1008,16553,16489,2885]
    Val_type4=[4023,16242,10587,13718,12974,13634,13721,10763,13817,2253,13544,18257,16940,17115,1712]
    Val_type5 = [9481,9567,12262,10089,9305,10002,12088,12001,9550,9565,9913,9653,9463,9915,9219]
    Val_type6 = [1744,17288,9378,9218,1830,17201,6149,9379,9291,15500,5747,13812,1012,3927,5834,]
    Val_type7 = [12019,5342,13812,6064,17506,17857,1919,12090,1834,9914,12020,5717,12004,5270,5343,]
    Val_type8 = [12099,18345,16490,5863,8043,9392,13369,18432,1783,15412,12012,5346,10847,13814,13806]
    Val_type9 = [6148,18396,6127,5802,5714,12179,5952,5715,14116,5889,6124,18395,8153,6064,6062,]
    Val_type10 = [12018,12369,834,9311,13419,11931,12282,14117,14028,6153,13504,12281,12095,9399,14203,]
    Val_type11 = [17288,4025,4024,3147,1012,1016,12325,10297,14494,14493,12361,18396,4020,3060,929,]
    Val_type12 = [5979,1966,9858,5892,1964,17288,4025,4024,3147,1012,1016,12325,10297,14494,14493,]
    Val_type15 = [18345,5892,1104,9861,5803,5979,14320,13812,9478,5717,17284,16614,9288,5801,9391,]
    Val_type16 = [1008,12627,9398,12107,11317,10208,9485,6793,9140,12190,6152,1976,10121,11315,12095,]
    Val_collection = [Val_type1, Val_type2, Val_type4, Val_type5, Val_type6, Val_type7, Val_type8,
                        Val_type9, Val_type10,
                        Val_type11, Val_type12, Val_type15, Val_type16]

    traindataset = Dataset(datadir, tileids="tileids/train_fold0.tileids")
    valdataset = Dataset(datadir, tileids="tileids/test_fold0.tileids")
    Trainsamples=[]
    for i in range(len(Train_collection)):
        Trainsamples = Trainsamples+Train_collection[i][:3*2]

    traindataset.samples = list(str(i) for i in Trainsamples)

    Valsamples = []
    for i in range(len(Val_collection)):
        Valsamples = Valsamples + Val_collection[i][:3 * 2]

    valdataset.samples = list(str(i) for i in Valsamples)

    traindataset.len = len(traindataset.samples)  # 39#351#3485（50Per）
    valdataset.len = len(valdataset.samples)  # 39#351#10929（50Per)

    traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize,shuffle=True,num_workers=workers,pin_memory=True)
    valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=batchsize, shuffle=False, num_workers=workers,
                                                  pin_memory=True)


    logger = Logger(columns=["loss"], modes=["train", "test"])

    vizlogger = VisdomLogger()

    network = UTAEClassification(UTAE(n_channels=13, n_classes=18, bilinear=True,
                                      encoder_widths=[ 32, 256],
                                      decoder_widths=[ 32, 256],
                                      agg_mode="att_mean",
                                      n_head=8,
                                      d_model=256,
                                      d_k=32))


    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss = torch.nn.NLLLoss()

    #weigths for class balance
    weights=1-torch.tensor([1,0.057328674,0.038091461,0.026397314,0.089135385,0.070171719,0.050900313,0.049949736,0.033796768,0.028428402,0.085305723,0.041688607,0.066786572,0.124587115,0.02854466,0.040806417,0.066854958,0.101226176])
    # print(len(weights))
    # lossC=torch.nn.CrossEntropyLoss(weight=weights)
    lossC = torch.nn.CrossEntropyLoss(ignore_index=0)


    start_epoch = 0

    if pretrain_pth is not None:
        print("Loading pre-trained model parameters...")
        # bert_path = os.path.join(pretrain_pth,  "checkpoint_"+str(50)+"."+ptr_model+".tar")
        bert_path=pretrain_pth
        if os.path.exists(bert_path):
            try:
                network.utae.load_state_dict(torch.load(bert_path))
            except:
                checkpoint = torch.load(bert_path)
                checkpoint_model = checkpoint
                state_dict = network.utae.state_dict()
                for k in ['inc.weight', 'inc.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

            for name, param in network.utae.named_parameters():
                print(name, param)
                break
            print('loaded',bert_path)

        else:
            print('Cannot find the pre-trained parameter file, please check the path!')


    if snapshot is not None:
        checkpoint = torch.load(snapshot)#, map_location=torch.device('cuda:0')
        start_epoch = checkpoint['epoch']
        network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network,device_ids=deviceID).cuda()
        lossC = lossC.cuda()

    writer = SummaryWriter(
        run_path)

    scaler = GradScaler()
    mini_acc = 0
    for epoch in range(start_epoch, epochs):

        logger.update_epoch(epoch)

        print("\nEpoch {}".format(epoch))
        print("train")
        Loss_list = 0
        Loss_list1= train_epoch(traindataloader, network, optimizer,lossC, (logger,vizlogger), Loss_list)

        writer.add_scalars("Training loss",
                           {'_time': Loss_list1},
                           global_step=epoch)

        Loss_list2 = val_epoch(valdataloader, network, optimizer, lossC, (logger, vizlogger), Loss_list,
                                )

        writer.add_scalars("Val loss",
                           {'_time': Loss_list2},
                           global_step=epoch)


        data = logger.get_data()
        vizlogger.update(data)

        if mini_acc < Loss_list2:
            mini_acc = Loss_list2
            save(epoch, network, optimizer, checkpoint_dir)

def train_epoch(dataloader, network, optimizer,lossC, loggers,Loss_list ):
    logger, vizlogger = loggers

    printer = Printer(N=len(dataloader))
    logger.set_mode("train")
    network.train()
    for iteration, data in enumerate(dataloader):

        optimizer.zero_grad()

        input, target = data


        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output,x2 = network.forward(input,torch.tensor(range(input.shape[1])).unsqueeze(0).repeat(input.shape[0],1).cuda())

        l=lossC(output,target)

        stats = {"loss":l.data.cpu().numpy()}

        l.backward()
        optimizer.step()

        printer.print(stats, iteration)

        Loss_list += l.item()

    return Loss_list

def val_epoch(dataloader, network, optimizer,lossC, loggers,Loss_list ):
    logger, vizlogger = loggers

    printer = Printer(N=len(dataloader))
    logger.set_mode("train")
    network.eval()

    firstIter=True
    with torch.no_grad():
        for iteration, data in enumerate(dataloader):
            optimizer.zero_grad()

            input, target = data
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            output,_ = network.forward(input,torch.tensor(range(input.shape[1])).unsqueeze(0).repeat(input.shape[0],1).cuda())

            output=output.argmax(dim=1)

            if firstIter:
                y_true = target.cpu().detach().numpy()
                y_pred = output.cpu().detach().numpy()
                firstIter = False
            else:
                y_true = np.concatenate((target.cpu().detach().numpy(), y_true), 0)
                y_pred = np.concatenate((output.cpu().detach().numpy(), y_pred), 0)
        y_true_=y_true[y_true>0]
        y_pred=y_pred[y_true>0]
        y_true=y_true_
        acc=accuracy_score(y_true.flatten(), y_pred.flatten())

    return acc


def save( epoch, model, optimizer,path):
    if not os.path.exists(path):
        os.makedirs(path)

    output_path = os.path.join(path, "model_best.tar")
    # output_path = os.path.join(path, "model_{:02d}.pth".format(epoch))
    torch.save({
        'epoch': epoch,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, output_path)


    print("EP:%d Model Saved on:" % epoch, output_path)


if __name__ == "__main__":

    args = parse_args()

    main(
        datadir = args.datadir,
        batchsize=args.batchsize,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.learning_rate,
        pretrain_pth=args.pretrain_pth,
        snapshot=args.snapshot,
        checkpoint_dir=args.checkpoint_dir,
        run_path=args.run_path
    )
