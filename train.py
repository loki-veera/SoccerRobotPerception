# -*- coding: utf-8 -*-
import argparse

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
import torchvision.utils as tutils

from utils.dataloader import (
    blobDataset,
    SegDataset,
    blob_dataloader,
    segmentation_dataloader,
)
from utils.model import nimbrRoNet2
from utils.metrics import metrics
from utils.losses import losses

# Argument Parser
parser = argparse.ArgumentParser(description="Nimbronet training")
parser.add_argument(
    "--epochs", default=300, type=int, help="number of total epochs to run"
)
parser.add_argument(
    "-b", "--batch_size", default=8, type=int, help="mini-batch size (default: 8)"
)
parser.add_argument(
    "--lr", "--learning-rate", default=0.001, type=float, help="initial learning rate"
)
parser.add_argument(
    "--blob_dir",
    default="/scratch/lveera2s/cudavision_data/blob/",
    help="Directory for blob images",
)
parser.add_argument(
    "--seg_dir",
    default="/scratch/lveera2s/cudavision_data/segmentation/",
    help="Directory for blob images",
)
parser.set_defaults(augment=True)
args = parser.parse_args()
print(args, flush=True)

# Summary Writer
writer = SummaryWriter(comment="_lr_" + str(args.lr) + "_Final")


def main():
    """Meta function to train the nimbronet
    """
    # Transforms
    transfs = transforms.Compose(
        [
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Dataloader
    trb_dataloader, valb_dataloader, testbdataloader, len_blob = blob_dataloader(
        args.blob_dir, transfs, args
    )
    trs_dataloader, vals_dataloader, testsdataloader, len_seg = segmentation_dataloader(
        args.seg_dir, transfs, args
    )
    counter = len_blob // len_seg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)

    # Model Instantiation
    nnet2 = nimbrRoNet2()
    nnet2.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    x for (name, x) in nnet2.named_parameters() if "resnet" not in name
                ]
            },
            {
                "params": [
                    x for (name, x) in nnet2.named_parameters() if "resnet" in name
                ],
                "lr": 1e-6,
            },
        ],
        lr=args.lr,
        betas=(0.8, 0.999),
    )

    print("Optimizer is done...", flush=True)

    # Training over epochs
    for epoch in range(args.epochs):
        # Train for one epoch
        btrloss, strloss = train(
            nnet2,
            trb_dataloader,
            trs_dataloader,
            optimizer,
            counter,
            epoch,
            writer,
            device,
        )

        # Validate over the epoch
        bvalloss, svalloss = validate(
            nnet2, valb_dataloader, vals_dataloader, epoch, writer, device
        )

        # Calculate metrics over validation set
        metrics.evaluate_blob(nnet2, valb_dataloader, device)
        print("**********")
        metrics.evaluate_seg(nnet2, vals_dataloader, device)

        # Save the model
        if (epoch + 1) % 20 == 0:
            torch.save(
                nnet2.state_dict(),
                "./checkpoints/models/Final_model_%03d.pt" % (epoch + 1),
            )
        print()
        print(
            "Epoch[{}/{}]: Blob training loss: {}: Segmentation training loss: {}".format(
                epoch + 1, args.epochs, btrloss, strloss
            ),
            flush=True,
        )
        print(
            "Epoch[{}/{}]: Blob validation loss: {}: Segmentation validation loss: {}".format(
                epoch + 1, args.epochs, bvalloss, svalloss
            ),
            flush=True,
        )

    # Evaluate over test set
    print("Evaluation: ")
    print("*************")
    metrics.evaluate_blob(nnet2, testbdataloader, device, writer)
    print()
    print("*************")
    print()
    metrics.evaluate_seg(nnet2, testsdataloader, device, writer)


def validate(nnet2, valb_loader, vals_loader, epoch, writer, device):
    """Method to perform validation over the epoch

    Args:
        nnet2: Nimbronet model
        valb_loader : Blob validation dataloader
        vals_loader: Segmentation validation dataloader
        epoch (int): Current epoch value
        writer (SummaryWriter): Summary writer for tensorboard
        device (str): String stating the device either GPU or CPU

    Returns:
        [float]: Blob validation loss
        [float]: Segmentation validation loss
    """
    nnet2.eval()
    val_blob_loss = 0
    val_seg_loss = 0
    with torch.no_grad():
        #Valdiate over blob
        for bindex, (images, targets) in enumerate(valb_loader):
            images = images.to(device)
            btargets = targets.to(device)
            _, blob_pred = nnet2(images)
            loss_blob = losses.detect_loss(blob_pred, btargets)
            val_blob_loss += loss_blob.item()
        # Validate over segmentation
        for sindex, (images, targets) in enumerate(vals_loader):
            images = images.to(device)
            targets = targets.to(device)
            seg_pred, _ = nnet2(images)
            loss_seg = losses.segment_loss(seg_pred, targets, device)
            val_seg_loss += loss_seg.item()
    val_blob_loss /= bindex + 1
    val_seg_loss /= sindex + 1
    seg_pred = seg_pred.cpu()
    seg_pred_1ch = torch.argmax(seg_pred, dim=1, keepdim=True)
    tutils.save_image(
        blob_pred.detach(),
        "./checkpoints/Adam_images/Final_blobepoch_%03d.png" % (epoch + 1),
        normalize=True,
    )
    tutils.save_image(
        btargets.detach(),
        "./checkpoints/Adam_images/Final_targetepoch_%03d.png" % (epoch + 1),
        normalize=True,
    )
    tutils.save_image(
        seg_pred_1ch.float(),
        "./checkpoints/Adam_images/Final_segepoch_%03d.png" % (epoch + 1),
        normalize=True,
    )

    writer.add_scalar("Blob_Valid_Loss", val_blob_loss, epoch)
    writer.add_scalar("Seg_Valid_Loss", val_seg_loss, epoch)

    return val_blob_loss, val_seg_loss


def train(
    nnet2, trb_dataloader, trs_dataloader, optimizer, counter, epoch, writer, device
):
    """Method to train over one epoch

    Args:
        nnet2: Nimbronet model
        trb_dataloader : Blob training dataloader
        trs_dataloader : Segmentation training dataloader
        optimizer : Optimizer used
        counter : Integer value for training the segmentation head
        epoch (int): Current epoch value
        writer (SummaryWriter): Summary writer for tensorboard
        device (str): String stating the device either GPU or CPU

    Returns:
        [float]: Blob training loss
        [float]: Segmentation training loss
    """
    nnet2.train()
    epoch_blob_loss = 0
    epoch_seg_loss = 0
    for index, (images, targets) in enumerate(trb_dataloader):
        optimizer.zero_grad()
        images = images.to(device)
        targets = targets.to(device)
        _, blob_pred = nnet2(images)
        loss_blob = losses.detect_loss(blob_pred, targets)
        epoch_blob_loss += loss_blob.item()
        loss_blob.backward()
        optimizer.step()

        if index % counter == 0:
            simages, stargets = next(iter(trs_dataloader))
            optimizer.zero_grad()
            simages = simages.to(device)
            stargets = stargets.to(device)
            seg_pred, _ = nnet2(simages)
            loss_seg = losses.segment_loss(seg_pred, stargets, device)
            epoch_seg_loss += loss_seg.item()
            loss_seg.backward()
            optimizer.step()
    epoch_blob_loss = epoch_blob_loss / (index + 1)
    epoch_seg_loss = epoch_seg_loss / ((index % counter) + 1)

    writer.add_scalar("Blob_Train_Loss", epoch_blob_loss, epoch)
    writer.add_scalar("Seg_Train_Loss", epoch_seg_loss, epoch)
    return epoch_blob_loss, epoch_seg_loss


if __name__ == "__main__":
    main()
