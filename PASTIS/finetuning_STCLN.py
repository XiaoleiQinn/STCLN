import sys
sys.path.append('/data/xiaolei.qin/Projects/mtlcc/utae-paps-main')
from src.dataset import PASTIS_Dataset
from src import utils
import numpy as np
import torch.nn
from torch.cuda.amp import autocast as autocast,GradScaler
from STCLN import UTAE,UTAEClassification

from logger import Logger, Printer, VisdomLogger
import argparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, cohen_kappa_score
import torch.nn.functional as F
from torch.autograd import Variable
from Models.focal_dice_loss import WDiceLossV2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
deviceID=[0]
torch.manual_seed(3407)
torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', "--datadir", default = "/data/xiaolei.qin/Projects/mtlcc/MTLCC-pytorch-master/src/data",type=str, help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=2 , type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=1, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=100, type=int, help="epochs to train" )
    parser.add_argument('-l', "--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument('-p', "--pretrain_pth", default="/data/xiaolei.qin/Projects/mtlcc/checkpoints/Pastis/SSpv/pretrain/UTA/batchtimespace0d4_fold5_32Is/checkpoint_99.sttrans.tar", type=str, help="pretrain path")
    parser.add_argument('-s', "--snapshot", default=None, type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default="/data/xiaolei.qin/Projects/mtlcc/checkpoints/Pastis/SSpv/STCLN_export", type=str, help="directory to save checkpoints")
    parser.add_argument('-runp', "--run_path", default="/data/xiaolei.qin/Projects/mtlcc/runs/Pastis/SSPV/finetune_STCLN_export/2Per_32Is/", type=str, help="run path")
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

    # ---------------pastis dataset-----------------------------
    collate_fn = lambda x: utils.pad_collate(x, pad_value=0)
    traindataset = PASTIS_Dataset(folder='/data/xiaolei.qin/Projects/CropClassification/Data/PASTIS/Data/PASTIS',
                                  norm=True,
                                  target='semantic', folds=[1])
    print('original len',len(traindataset))
    traindataset.id_patches = [i for i in
                               [10279, 40282, 10174, 40538, 10457, 30088, 10151, 30602, 40275, 20052, 20153, 10056,
                                20147, 40420, 10450, 20614, 20167, 40446, 20251, 20103, 20613, 20349, 20384, 40337,
                                40163, 10030, 10068, 10399, 30327, 10289, 30351, 40198, 20167, 20153, 20207, 20203,
                                10000, 10354, 20214, 10007, 10392, 30055, 10054, 20009, 10040, 10147, 10110, 10105,
                                20398, 20469, 20203, 20254, 30402, 30013, 40200, 30689, 20115, 20356, 20345, 20466,
                                40033, 10153, 30153, 30012, 40340, 30140, 40214, 10129, 30211, 20078, 30109, 30013,
                                20116, 20243, 20085, 40271]]
    traindataset.len = len(traindataset.id_patches)
    print(traindataset.len)
    valdataset = PASTIS_Dataset(folder='/data/xiaolei.qin/Projects/CropClassification/Data/PASTIS/Data/PASTIS',
                                norm=True,
                                target='semantic', folds=[2])

    valdataset.id_patches = [i for i in
                             [10127, 40438, 30643, 30619, 30436, 10306, 10127, 30226, 40027, 40279, 40085, 40455, 40426,
                              40444, 40298, 40231, 20498, 40298, 20149, 20315, 20340, 20618, 20412, 20231, 40006, 40138,
                              40186, 10087, 40430, 30194, 10247, 40383, 20364, 20128, 20060, 20118, 40001, 20080, 40063,
                              30000, 40063, 40000, 40348, 10021, 40001, 20023, 30000, 20441, 20235, 20344, 20297, 20576,
                              30014, 30275, 30298, 30083, 20206, 20443, 20423, 20380, 30064, 30084, 30145, 10107, 10225,
                              40558, 10383, 20131, 30010, 20022, 30295, 20042, 20031, 40046, 20358, 30142]]
    valdataset.len = len(valdataset.id_patches)

    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True, num_workers=workers,
                                                  pin_memory=True, collate_fn=collate_fn)
    valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=batchsize, shuffle=False, num_workers=workers,
                                                pin_memory=True, collate_fn=collate_fn)

    logger = Logger(columns=["loss"], modes=["train", "test"])

    vizlogger = VisdomLogger()
    network = UTAEClassification(UTAE(n_channels=10, n_classes=20, bilinear=True,
                                      encoder_widths=[ 32, 256],
                                      decoder_widths=[ 32, 256],
                                      agg_mode="att_mean",
                                      n_head=8,
                                      d_model=256,
                                      d_k=32))


    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    weights = torch.ones(20).float()
    weights[0] = 0
    weights[19]=0

    lossC=torch.nn.CrossEntropyLoss(weight=weights)
    loss=WDiceLossV2(apply_nonlin=F.softmax)
    start_epoch = 0

    if pretrain_pth is not None:
        print("Loading pre-trained model parameters...")
        bert_path=pretrain_pth
        if os.path.exists(bert_path):
            network.utae.load_state_dict(torch.load(bert_path))
            for name, param in network.utae.named_parameters():
                print(name, param)
                break
            print('loaded',bert_path)

        else:
            print('Cannot find the pre-trained parameter file, please check the path!')


    if snapshot is not None:
        checkpoint = torch.load(snapshot)
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
        Loss_list1= train_epoch(traindataloader, network, optimizer,lossC, (logger,vizlogger), Loss_list,scaler)

        writer.add_scalars("Training loss",
                           {'_time': Loss_list1},
                           global_step=epoch)

        Loss_list2 = val_epoch(valdataloader, network, optimizer, lossC, (logger, vizlogger), Loss_list,
                                  scaler)

        writer.add_scalars("Val loss",
                           {'_time': Loss_list2},
                           global_step=epoch)



        data = logger.get_data()
        vizlogger.update(data)

        if mini_acc < Loss_list2:
            mini_acc = Loss_list2
            save(epoch, network, optimizer, checkpoint_dir)

def train_epoch(dataloader, network, optimizer,lossC, loggers,Loss_list ,scaler):
    logger, vizlogger = loggers

    printer = Printer(N=len(dataloader))
    logger.set_mode("train")
    network.train()

    for iteration, data in enumerate(dataloader):
        for i,j in [[0,0],[1,1]]:#[[0,0],[1,1],[2,2],[3,3],[1,0]] 1->5Per
            optimizer.zero_grad()

            (input, dates), target = data

            split = input.shape[-1] // 4
            input = input[:, :, :, i * split:(i + 1) * split, j * split:(j + 1) * split]
            target = target[:, i * split:(i + 1) * split, j * split:(j + 1) * split]

            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            output,x2 = network.forward(input,torch.tensor(range(input.shape[1])).unsqueeze(0).repeat(input.shape[0],1).cuda())

            l=lossC(output,target)

            stats = {"loss":l.data.cpu().numpy()}

            l.backward()
            optimizer.step()

            printer.print(stats, iteration)
            logger.log(stats, iteration)
            vizlogger.plot_steps(logger.get_data())
            Loss_list += l.item()


    return Loss_list

def val_epoch(dataloader, network, optimizer,lossC, loggers,Loss_list ,scaler):
    logger, vizlogger = loggers

    printer = Printer(N=len(dataloader))
    logger.set_mode("train")
    network.eval()

    firstIter=True
    with torch.no_grad():
        for iteration, data in enumerate(dataloader):
            for i,j in [[0,0],[1,1]]:#[[0,0],[1,1],[2,2],[3,3],[1,0]]
                optimizer.zero_grad()

                # pastis
                (input, dates), target = data

                split = input.shape[-1] // 4
                input = input[:, :, :, i * split:(i + 1) * split, j * split:(j + 1) * split]
                target = target[:, i * split:(i + 1) * split, j * split:(j + 1) * split]

                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()

                output,x2 = network.forward(input,torch.tensor(range(input.shape[1])).unsqueeze(0).repeat(input.shape[0],1).cuda())

                output=output.argmax(dim=1)

                if firstIter:
                    y_true = target.cpu().detach().numpy()
                    y_pred = output.cpu().detach().numpy()
                    firstIter = False
                else:
                    y_true = np.concatenate((target.cpu().detach().numpy(), y_true), 0)
                    y_pred = np.concatenate((output.cpu().detach().numpy(), y_pred), 0)
        y_true_ = y_true[(y_true < 19) & (y_true > 0)]
        y_pred = y_pred[(y_true < 19) & (y_true > 0)]
        y_true = y_true_
        acc=accuracy_score(y_true.flatten(), y_pred.flatten())

    return acc


def save( epoch, model, optimizer,path):
    if not os.path.exists(path):
        os.makedirs(path)

    output_path = os.path.join(path, "model_best.tar")
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
