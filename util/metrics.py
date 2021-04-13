'''
@Author: Zhou Kai
@GitHub: https://github.com/athon2
@Date: 2018-11-03 09:47:14
'''
import torch
from torch.nn.modules.loss import _Loss 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _Loss, _WeightedLoss

class SoftDiceLoss(_WeightedLoss):
    """
    Dice Loss for a batch of samples
    """

    def forward(self, output, target, weights=None, binary=False):
        """
        Forward pass
        :param output: NxCxHxW logits
        :param target: NxHxW LongTensor
        :param binary: bool for binarized one chaneel(C=1) input
        :return: torch.tensor
        """
        output = F.softmax(output, dim=1)
        #print(f'output shape {output.shape}')
        #print(f'target {target.shape}')
        (unique, counts) = np.unique(target.cpu(), return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        #print(f'frequency {frequencies}')
        #print(output)
        #print(target)
        if binary:
            return self._dice_loss_binary(output, target)
        return self._dice_loss_multichannel(output, target, weights)

    @staticmethod
    def _dice_loss_binary(output, target):
        """
        Dice loss for one channel binarized input
        :param output: Nx1xHxW logits
        :param target: NxHxW LongTensor
        :return:
        """
        eps = 0.0001
        intersection = output * target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + target
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = 1 - (numerator / denominator)

        return loss_per_channel.sum() / output.size(1)

    @staticmethod
    def _dice_loss_multichannel(output, target, weights=None):
        """
        Forward pass
        :param output: NxCxHxW Variable
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param binary: bool for binarized one chaneel(C=1) input
        :return:
        """

        #output = F.softmax(output, dim=1)
        eps = 0.0001
        #target = target.unsqueeze(1)
        encoded_target = torch.zeros_like(output)

        encoded_target = encoded_target.scatter(1, target, 1)
        #print(target)
        #print(encoded_target)
        #print(encoded_target.shape)
        intersection = output * encoded_target
        #print(intersection.shape)
        intersection = intersection.sum(2).sum(2)
        #print(intersection)

        num_union_pixels = output + encoded_target
        num_union_pixels = num_union_pixels.sum(2).sum(2)

        loss_per_class = 1 - ((2 * intersection) / (num_union_pixels + eps))
        # loss_per_class = 1 - ((2 * intersection + 1) / (num_union_pixels + 1))
        if weights is None:
            weights = torch.ones_like(loss_per_class)
            #weights[:,0,:] = weights[:,0,:]*0
        loss_per_class *= weights
        #print(weights.shape, loss_per_class.shape)
        #print(loss_per_class)
        return (loss_per_class.sum(1) / (num_union_pixels != 0).sum(1).float()).mean()

class SoftDiceLoss_v2(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division, 
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, output, target, weights=None, binary=True):
        """
        Forward pass
        :param output: NxCxHxW logits
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param binary: bool for binarized one chaneel(C=1) input
        :return: torch.tensor
        """
        #print(f'output shape {output.shape}')
        #print(f'target {target.shape}')
        #print(output)
        #print(target)
        output = F.softmax(output, dim=1)
        #print(output)
        if binary:
            return self._dice_loss_binary(output, target)
        return self._dice_loss_multichannel(output, target, weights)

    @staticmethod
    def _dice_loss_binary(output, target):
        """
        Dice loss for one channel binarized input
        :param output: Nx1xHxW logits
        :param target: NxHxW LongTensor
        :return:
        """
        eps = 0.0001

        intersection = output * target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + target
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = 1 - (numerator / denominator)

        return loss_per_channel.sum() / output.size(1)

   # def forward(self, y_pred, y_true, eps=1e-8):
   #     y_pred = F.softmax(y_pred, dim=1)
   #     intersection = torch.sum(torch.mul(y_pred, y_true)) 
   #     union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps

   #     dice = 2 * intersection / union 
   #     dice_loss = 1 - dice

   #     return dice_loss

class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(torch.log(torch.mul(std, std))) - 1
    
class CombinedLoss(_Loss):
    '''
    Combined_loss = Dice_loss + k1 * L2_loss + k2 * KL_loss
    As default: k1=0.1, k2=0.1
    '''
    def __init__(self, k1=0.1, k2=0.1):
        super(CombinedLoss, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()

    def forward(self, y_pred, y_true, y_mid):
        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        seg_pred, seg_truth =  (y_pred[:,0,:,:,:], y_true[:,0,:,:,:]) #training each label seperately  Ben
        vae_pred, vae_truth = (y_pred[:, 1:, :, :, :], y_true[:, 1:, :, :, :])
        # seg_pred, seg_truth = (y_pred[:, 0:3, :, :, :], y_true[:, 0:3, :, :, :]) #training 3 labels together
        # vae_pred, vae_truth = (y_pred[:, 3:, :, :, :], y_true[:, 3:, :, :, :])
        dice_loss = self.dice_loss(seg_pred, seg_truth)
        l2_loss = self.l2_loss(vae_pred, vae_truth)
        kl_div = self.kl_loss(est_mean, est_std)
        combined_loss = dice_loss + self.k1 * l2_loss + self.k2 * kl_div
        #print("dice_loss:%.4f, L2_loss:%.4f, KL_div:%.4f, combined_loss:%.4f"%(dice_loss,l2_loss,kl_div,combined_loss))
        
        return combined_loss
