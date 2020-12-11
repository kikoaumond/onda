import torch
torch.nn.BCEWithLogitsLoss


class MostDiscriminantFeatureLoss(torch.nn.Module):

    def __init__(self, reduction=None):
        """
        Constructor for  MostDiscriminantFeatureLoss

        Args:
            reduction (str): one of 'mean', 'sum' or 'None'; specified if reduction will be applied
                to the loss and if so, how.
                Dwfault: None
        """
        super(MostDiscriminantFeatureLoss, self).__init__()
        self._reduction = reduction

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss