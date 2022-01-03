import torch
from torch.nn import MSELoss, CrossEntropyLoss, NLLLoss


class losses:
    def __init__(self):
        pass

    def total_variation_loss(self, x):
        """Source: https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/\
    tensorflow/python/ops/image_ops_impl.py#L3085-L3154"""
        pixel_diff1 = abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        pixel_diff2 = abs((x[:, :, :, 1:] - x[:, :, :, :-1]))
        return (torch.sum(pixel_diff1) + torch.sum(pixel_diff2)) / x.shape[0]

    @staticmethod
    def detect_loss(self, blob_preds, targets):
        """Method to calculate the weighted detection loss

        Args:
            blob_preds (Tensor): Predictions from the model
            targets (Tensor): Target images

        Returns:
            [float]: calculated loss value 
        """

        mse = MSELoss()
        # tv_loss = total_variation_loss(blob_preds)*0.00000002
        tv_loss = self.total_variation_loss(blob_preds) * 0.000002
        # tv_loss = total_variation_loss(blob_preds)*0.000002
        # mse_loss = mse(blob_preds,targets)*0.3
        mse_loss = mse(blob_preds, targets)
        return tv_loss + mse_loss

    @staticmethod
    def segment_loss(self, seg_preds, targets, device):#
        """Method to calculate the weighted segmentation loss

        Args:
            seg_preds (Tensor):  Predictions from the model
            targets (Tensor): Target images
            device (str): string holding device information (GPU or CPU)

        Returns:
            [float]: Calculated loss value
        """
        weights = [0.6, 0.5, 0.95]
        class_weights = torch.FloatTensor(weights).cuda()
        cross_entropy = CrossEntropyLoss(weight=class_weights)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)
        loss_ce = cross_entropy(seg_preds, targets)
        x = seg_preds[:, [0, 1], :, :]
        loss_tvar = self.total_variation_loss(x)

        return loss_ce + 0.00003 * loss_tvar
