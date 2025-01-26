import sys
sys.path.append('/data/xiaolei.qin/Projects/mtlcc/utae-paps-main')
from src.dataset import PASTIS_Dataset
from src import utils
from torch.cuda.amp import autocast as autocast,GradScaler
import numpy as np
import torch.nn

from STCLN import UTAE,UTAEPrediction

from logger import Logger, Printer, VisdomLogger
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
deviceIds=[0,1,2,3]
from torch.utils.data.dataset import ConcatDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', "--datadir", default = "/data/xiaolei.qin/Projects/CropClassification/Data/PASTIS/Data/PASTIS",type=str, help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=4, type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=8, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=100, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default=None, type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default="/data/xiaolei.qin/Projects/mtlcc/checkpoints/Pastis/SSpv/pretrain/export_ptr", type=str, help="directory to save checkpoints")
    return parser.parse_args()


def main(
    datadir,
    batchsize = 16,
    workers = 8,
    epochs = 100,
    lr = 1e-3,
    snapshot = None,
    checkpoint_dir = None,
    ptr_model = None
    ):

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Pastis
    collate_fn = lambda x: utils.pad_collate(x, pad_value=0)
    traindataset = PASTIS_Dataset(folder=datadir,
                                  norm=True,
                                  target='semantic', folds=[5])

    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True, num_workers=workers,
                                                  pin_memory=True, collate_fn=collate_fn)


    logger = Logger(columns=["loss"], modes=["train", "test"])

    vizlogger = VisdomLogger()

    network = UTAEPrediction(UTAE(n_channels=10, n_classes=20, bilinear=True,
                                      encoder_widths=[32, 256],
                                      decoder_widths=[32, 256],
                                      agg_mode="att_mean",
                                      n_head=8,
                                      d_model=256,
                                      d_k=32),num_features=10,dropout=0.4)


    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss = torch.nn.MSELoss(reduction='none')

    start_epoch = 0

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
        network = torch.nn.DataParallel(network,device_ids=deviceIds).cuda()

        loss = loss.cuda()

    writer = SummaryWriter(
        '/data/xiaolei.qin/Projects/mtlcc/runs/Pastis/SSPV/Pretrain/'+'_STCLN')#_deepsup_timespace_check

    scaler = GradScaler()
    mini_loss = np.Inf
    for epoch in range(start_epoch, epochs):

        logger.update_epoch(epoch)

        print("\nEpoch {}".format(epoch))
        print("train")
        Loss_list = 0
        Loss_list1= train_epoch(traindataloader, network, optimizer, loss, (logger,vizlogger), Loss_list,scaler)
        if epoch!=start_epoch:
            writer.add_scalars("Training loss",
                           {'time': Loss_list1},#SSpv_kmeans_
                           global_step=epoch)

        data = logger.get_data()
        vizlogger.update(data)

        if (epoch % 40 == 0 or epoch + 1 == args.epochs):
            save(epoch, network, optimizer, checkpoint_dir,ptr_model)

def train_epoch(dataloader, network, optimizer, loss, loggers,Loss_list,scaler):
    logger, vizlogger = loggers

    printer = Printer(N=len(dataloader))
    logger.set_mode("train")

    network.train()

    for iteration, data in enumerate(dataloader):
        for i in range(4):
            for j in range(4):
                optimizer.zero_grad()

                # Pastis
                (input, dates), label = data

                input = input.cuda()
                split = input.shape[-1] // 4
                input = input[:, :, :, i * split:(i + 1) * split, j * split:(j + 1) * split]

                cluster_ids_x = (input[:, :, 6, :, :] - input[:, :, 2, :, :]) / (
                            input[:, :, 2, :, :] + input[:, :, 6, :, :] + 1e-20)
                cluster_ids_x = cluster_ids_x.gt(0.2).int().unsqueeze(2).repeat(1, 1, input.shape[2], 1, 1)


                output,x2, target, mask = network.forward(input,cluster_ids_x,torch.tensor(range(input.shape[1])).unsqueeze(0).repeat(input.shape[0],1).cuda())

                l = loss(output, target.float())#+loss(x2,target.float())

                l = (l * (1-mask).float()).sum() / ((1-mask).sum()+1e-20)

                stats = {"loss":l.data.cpu().numpy()}

                l.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()


                printer.print(stats, iteration)
                logger.log(stats, iteration)
                vizlogger.plot_steps(logger.get_data())
                Loss_list += l.item()

    return Loss_list


def save( epoch, model, optimizer,path,ptr_model):
    if not os.path.exists(path):
        os.makedirs(path)

    output_path = os.path.join(path, "checkpoint_"+str(epoch)+".tar")
    torch.save({
        'epoch': epoch,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, output_path)

    bert_path = os.path.join(path, "checkpoint_"+str(epoch)+"."+ptr_model+".tar")

    torch.save(model.module.utae.state_dict(), bert_path)

    print("EP:%d Model Saved on:" % epoch, output_path,bert_path)


if __name__ == "__main__":

    args = parse_args()

    main(
        datadir = args.datadir,
        batchsize=args.batchsize,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.learning_rate,
        snapshot=args.snapshot,
        checkpoint_dir=args.checkpoint_dir
    )
