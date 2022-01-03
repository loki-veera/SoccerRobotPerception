import torch
from torch.utils.data import DataLoader, Dataset
import glob
from PIL import Image
from skimage import io
import os
import xml.etree.ElementTree as ET
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.utils as utils
import os.path as osp


class blobDataset(Dataset):
    def __init__(self, blobdir, transfs=None):
        self.image_list = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            self.image_list.extend(sorted(glob.glob(blobdir + "/" + ext)))
        # Remove files that don't have an annotation file
        self.image_list = list(
            filter(lambda x: osp.isfile(osp.splitext(x)[0] + ".xml"), self.image_list)
        )

        self.objects = {"ball": 0, "goalpost": 1, "robot": 2}
        self.variance = {"ball": 5.0, "goalpost": 5.0, "robot": 10.0}
        self.transforms = transfs

    def __getitem__(self, index):
        # Read the indexed image
        image = Image.fromarray(io.imread(self.image_list[index])).convert("RGB")
        image = image.resize((640, 480), Image.ANTIALIAS)
        annotation_file = osp.splitext(self.image_list[index])[0] + ".xml"

        size = image.size
        # Parsing the annotaion XML file
        tree = ET.parse(annotation_file)
        h, w = 480 // 4, 640 // 4
        target = np.ones((len(self.objects), h, w), dtype="float32")
        coords = np.dstack((np.mgrid[0:h, 0:w]))
        # Parser tree iterator to get objects, tag and their values
        for objects in tree.iter():
            if objects.tag == "object":
                for obj in objects:
                    if obj.tag == "name":
                        label = obj.text
                    if obj.tag == "bndbox":
                        bbox = []
                        for val in obj:
                            bbox.append(float(val.text))

                        # Centroid for boudning boxes
                        mid_val = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                        var = self.variance[label]
                        mu = [mid_val[1] * h / size[1], mid_val[0] * w / size[0]]

                        # Get gaussian pdf for the mean and variance values
                        gauss_pdf = multivariate_normal(mu, cov=var)

                        # Impose on the target image
                        target[self.objects[label], :, :] -= (
                            var * var * gauss_pdf.pdf(coords).reshape(h, w)
                        )

        target = torch.from_numpy(target)
        # target /= (target.max()+1e-7)
        # target = transforms.ToTensor()(target)
        if self.transforms:
            image = self.transforms(image)
        return image, target

    def __len__(self):
        return len(self.image_list)


class SegDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.target_img_names = sorted(os.listdir(path + "/target/"))
        self.train_img_names = sorted(os.listdir(path + "/image/"))
        self.transforms = transforms
        self.path = path

    def __getitem__(self, index):
        # Read the indexed training and target image
        train_name = self.path + "/image/" + self.train_img_names[index]
        target_name = self.path + "/target/" + self.target_img_names[index]

        train_image = Image.open(train_name)
        target_image = Image.open(target_name)

        target_image = np.array(target_image)

        # Declaring the whole image as background
        target = np.zeros_like(target_image)

        # Index each individual object in each channel of the image
        target[target_image == 2] = 2  # line
        target[target_image == 1] = 1  # ball
        target[target_image == 3] = 1  # field

        resize = transforms.Resize((480 // 4, 640 // 4))
        target = resize(Image.fromarray(target))
        # target = np.resize()
        target = np.array(target)
        target = torch.FloatTensor(target)

        # Apply image transformation
        image = train_image
        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.train_img_names)


def blob_dataloader(blob_dir, transfs, args):
    """Method to create dataloader for blob dataset

    Args:
        blob_dir (str): Path to the blob dataset
        transfs (transforms): transforms
        args (argument parser): argument parser
    """
    blobFolders = ["dataset", "forceTest", "forceTrain"]
    blobDatasets = []
    for folder in blobFolders:
        blobDatasets.append(blobDataset(blob_dir + folder, transfs))
    total_blob_dataset = torch.utils.data.ConcatDataset(blobDatasets)
    train_blob, val_blob, test_blob = torch.utils.data.random_split(
        total_blob_dataset, [6200, 1329, 1329]
    )
    trb_dataloader = torch.utils.data.DataLoader(
        train_blob, batch_size=args.batch_size, num_workers=6, shuffle=True
    )  # 16
    valb_dataloader = torch.utils.data.DataLoader(
        val_blob, batch_size=args.batch_size, num_workers=6, shuffle=True
    )  # 16
    testb_dataloader = torch.utils.data.DataLoader(
        test_blob, batch_size=args.batch_size, num_workers=6, shuffle=True
    )  # 16
    return trb_dataloader, valb_dataloader, testb_dataloader, len(train_blob)


def segmentation_dataloader(seg_dir, transfs, args):
    """Method to create dataloader for segmentation dataset

    Args:
        seg_dir (str): Path to the segmentation dataset
        transfs (transforms): transforms
        args (argument parser): argument parser
    """
    segFolders = ["dataset", "forceTrain"]
    segDatasets = []
    for folder in segFolders:
        segDatasets.append(SegDataset(seg_dir + folder, transfs))
    total_seg_dataset = torch.utils.data.ConcatDataset(segDatasets)
    train_seg, val_seg, test_seg = torch.utils.data.random_split(
        total_seg_dataset, [835, 178, 179]
    )
    trs_dataloader = torch.utils.data.DataLoader(
        train_seg, batch_size=args.batch_size, num_workers=6, shuffle=True
    )  # 16
    vals_dataloader = torch.utils.data.DataLoader(
        val_seg, batch_size=args.batch_size, num_workers=6, shuffle=True
    )  # 16
    tests_dataloader = torch.utils.data.DataLoader(
        test_seg, batch_size=args.batch_size, num_workers=6, shuffle=True
    )  # 16
    return trs_dataloader, vals_dataloader, tests_dataloader, len(train_seg)
