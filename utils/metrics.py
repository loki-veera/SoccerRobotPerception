import numpy as np
import torch
import cv2


class metrics:
    def __init__(self):
        pass

    def extractCentroids(self, singleChannelImage):
        """Extract the centroids

        Args:
            singleChannelImage (Tensor): Single channel of the image

        Returns:
            [list]: List of centroids in the image
        """
        # Extract centroids from the images
        centroids = []
        _, thresh = cv2.threshold(singleChannelImage, 0.8, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.convertScaleAbs(thresh)
        thresh = cv2.blur(thresh, (5, 5))
        # Find contours and then moments from the contours to calculate centroids
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            M = cv2.moments(c)
            cX = int(M["m10"] / (M["m00"] + 0.001))
            cY = int(M["m01"] / (M["m00"] + 0.001))
            centroids.append([cX, cY])
        return centroids

    def evaluateBlobBatch(self, targets, predictions):
        """Evaluate the batch of images for blobs

        Args:
            targets (Tensor): Target images
            predictions (Tensor): Predicted images
        """
        # Calculate F1, Accuracy, Recall, Precision and False detection rate for each batch
        n_batch, _, _, _ = targets.shape
        f1 = [0, 0, 0]
        acc = [0, 0, 0]
        rec = [0, 0, 0]
        precision = [0, 0, 0]
        fdr = [0, 0, 0]
        # Threshold for the detection
        thresh_dist = 10
        for idx in range(n_batch):
            target = targets[idx, :, :, :].squeeze().detach().cpu().numpy()
            pred = predictions[idx, :, :, :].squeeze().detach().cpu().numpy()
            # For each class calculate TP, TN, FP, FN based on threshold distance
            # between the target centroids and predicted centroids.
            for cls in [0, 1, 2]:
                tp = tn = fp = fn = 0
                true_centroids = self.extractCentroids(target[cls, :, :])
                pred_centroids = self.extractCentroids(pred[cls, :, :])
                true_length = len(true_centroids)
                pred_length = len(pred_centroids)
                detected = np.zeros(true_length)
                # For each predicted centroid check in target if less than threshold
                # tp else fp and fn is where it didnt found in target
                # tn is if channel has no object and our image predicted none
                for k in range(pred_length):
                    found = -1
                    for l in range(true_length):
                        dist = np.linalg.norm(
                            np.asarray(true_centroids[l])
                            - np.asarray(pred_centroids[k])
                        )
                        if dist <= thresh_dist:
                            tp += 1
                            found = l
                            break
                    if found == -1:
                        fp += 1
                    else:
                        detected[found] = 1
                    fn = np.count_nonzero(detected == 0)
                if len(pred_centroids) == len(true_centroids) == 0:
                    tn += 1

                acc[cls] += (tp + tn) / ((tp + tn + fp + fn + 1e-6) * n_batch)
                rec[cls] += tp / ((tp + fn + 1e-6) * n_batch)
                precision[cls] += tp / ((tp + fp + 1e-6) * n_batch)
                f1[cls] += (2 * precision[cls] * rec[cls]) / (
                    (precision[cls] + rec[cls] + 1e-6) * n_batch
                )
                fdr[cls] += fp / ((tp + fp + 1e-6) * n_batch)
        return acc, rec, precision, f1, fdr

    @staticmethod
    def evaluate_blob(self, nnet2, loader, device, writer=None):
        """This method computes detection metrics over all the images in loader 

        Args:
            nnet2 : Nimbronet2 model
            loader : Specifiec dataloader
            device (śtr): string holding device information either GPU/CPU
            writer (Summary writer, optional): Summary writer. Defaults to None.
        """
        nnet2.eval()
        class_f1 = []
        class_acc = []
        class_rec = []
        class_precision = []
        class_fdr = []

        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            _, blob_pred = nnet2(images)
            acc, rec, precision, f1, fdr = self.evaluateBlobBatch(blob_pred, targets)
            class_acc.append(acc)
            class_rec.append(rec)
            class_precision.append(precision)
            class_f1.append(f1)
            class_fdr.append(fdr)

        class_f1 = np.asarray(class_f1)
        class_acc = np.asarray(class_acc)
        class_rec = np.asarray(class_rec)
        class_precision = np.asarray(class_precision)
        class_fdr = np.asarray(class_fdr)

        class_f1 = np.mean(class_f1, axis=0)
        class_acc = np.mean(class_acc, axis=0)
        class_rec = np.mean(class_rec, axis=0)
        class_precision = np.mean(class_precision, axis=0)
        class_fdr = np.mean(class_fdr, axis=0)

        if writer:
            print("CLASS FORMAT: [BALL, GOALPOST, ROBOT]")
            print("*************************************************")
            print("F1 score: ")
            print(class_f1)
            print("*************************************************")
            print("Accuracy: ")
            print(class_acc)
            print("*************************************************")
            print("Recall: ")
            print(class_rec)
            print("*************************************************")
            print("Precision: ")
            print(class_precision)
            print("*************************************************")
            print("False Detection rate: ")
            print(class_fdr)
            print("*************************************************")
        else:
            print(class_f1, class_acc, class_rec, class_precision, class_fdr)

    def evaluateSegBatch(self, targets, predictions):
        """Evaluate the batch of images for segmentation

        Args:
            targets (Tensor): Target images
            predictions (Tensor): Predicted images
        """
        # Calculate IOU, Accuracy on segmentation set
        # n_batch,_,_,_ = targets.shape
        n_batch, _, _ = targets.shape
        ious = [0, 0, 0]
        accs = [0, 0, 0]

        for idx in range(n_batch):
            # target = targets[idx,:,:,:].squeeze().cpu().numpy()
            # pred = predictions[idx,:,:,:].squeeze().detach().cpu().numpy()
            target = targets[idx, :, :].cpu().numpy()
            pred = predictions[idx, :, :].detach().cpu().numpy()

            # All two unique values were replaced with 1, 2.
            unique_val = np.unique(target)
            try:
                target[target == unique_val[-2]] = 1
                target[target == unique_val[-1]] = 2
            except:
                continue

            unique_val = np.unique(pred)
            # print(unique_val)
            pred[pred == unique_val[-2]] = 1
            pred[pred == unique_val[-1]] = 2
            target = target.astype(np.int)
            pred = pred.astype(np.int)

            # For each class calculate the IOU and Accuracy from Intersection
            # and Union of the area in predicted and target image
            for cls in [0, 1, 2]:
                pred_indices = pred == cls
                target_indices = target == cls
                intersection = np.sum(np.logical_and(pred_indices, target_indices))
                union = np.sum(np.logical_or(pred_indices, target_indices))
                ious[cls] += (intersection / (union + 0.001)) / n_batch
                accs[cls] += (intersection / (np.sum(target_indices) + 0.001)) / n_batch
                # print((intersection/(union+0.001))/n_batch, (intersection/(np.sum(target_indices)+0.001))/n_batch)
            # print(ious,accs)
        return ious, accs

    @staticmethod
    def evaluate_seg(self, nnet2, loader, device, writer=None):
        """This method computes segmenation metrics over all the images in loader 

        Args:
            nnet2 : Nimbronet2 model
            loader : Specific dataloader
            device (śtr): string holding device information either GPU/CPU
            writer (Summary writer, optional): Summary writer. Defaults to None.
        """
        class_acc = []
        class_iou = []
        nnet2.eval()
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            seg_preds, _ = nnet2(images)
            seg_preds = torch.argmax(seg_preds, dim=1)
            # print(targets.shape, seg_preds.shape)
            iou, acc = self.evaluateSegBatch(seg_preds, targets)
            class_iou.append(iou)
            class_acc.append(acc)
        class_iou = np.asarray(class_iou)
        class_acc = np.asarray(class_acc)
        c_iou = np.mean(class_iou, axis=0)
        c_acc = np.mean(class_acc, axis=0)

        if writer:
            print()
            print("IOU and Accuracy:")
            print("Accuracy (Field, line, BG): ", c_acc[1], c_acc[2], c_acc[0])
            print("IOU (Field, line, BG): ", c_iou[1], c_iou[2], c_iou[0])
            print()
        else:
            print(c_iou, c_acc)
