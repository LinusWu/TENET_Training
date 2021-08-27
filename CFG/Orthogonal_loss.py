# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

class OrthogonalLoss(nn.Module):
    """
        A module caculates Othogonal loss of grouped features.
        Input: 1.[batch,clsuters] tensor.
               2.An index array, corresponding to cluster label.
        Return: A orthogonal loss.
    """
    def __init__(self, num_clusters, device=torch.device('cuda')):
        super(OrthogonalLoss, self).__init__()
        self.num_clusters = num_clusters
        self.device = device


    def forward(self, x, cluster_ids):
        cluster_ids = cluster_ids.cpu().numpy()
        loss_total = 0
        batch_num = x.size(0)
        for batch in range(batch_num):
            stack_list = []
            # Some clusters might be null, which will cause NAN when caculate mean value of group.
            valid_num_clusters = self.num_clusters
            for index in range(valid_num_clusters):
                feature_pos = np.where(cluster_ids[batch] == index)
                if feature_pos[0].shape[0] == 0:
                    # null cluster
                    valid_num_clusters -= 1
                    continue
                tensor_group = x[batch][feature_pos]
                tensor_group_mean = tensor_group.mean(dim=0).view(-1)
                stack_list.append(tensor_group_mean)
            stack_matrix = torch.stack(stack_list)
            # get Orthogonal loss matrix
            Ort_loss_matrix = torch.abs(stack_matrix.mm(stack_matrix.t())) 
            num = valid_num_clusters * (valid_num_clusters-1)  / 2 if valid_num_clusters > 1 else 1
            loss_total += torch.sum(torch.triu(Ort_loss_matrix, diagonal=1)) / num
        return loss_total / batch_num