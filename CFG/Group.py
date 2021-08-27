# -*- coding: utf-8 -*-
import torch
import numpy as np
import time
import torch.nn.functional as F
from Orthogonal_loss import OrthogonalLoss

class Cluster_GPU_FAST():
    """"
    A class for clustering feature maps.
    Input:  1. A set of feature maps with 2-4 dimension.
            2. Number of cluster centers.
    Return: 1. An index array, corresponding to cluster label.
            2. Cluster center of each cluster.
            3. (Optional) Orthogonal loss caculated by each group of feature maps.
    """

    def __init__(self, num_clusters, shift_threshold=1e-4, max_iter=50, device=torch.device('cuda'), debug=False):
        self.cluster = KMeans_GPU_FAST(num_clusters=num_clusters, shift_threshold=shift_threshold, max_iter=max_iter, device=device)
        self.device = device
        self.debug = debug
        self.Func_Orth = OrthogonalLoss(num_clusters, device=device)


    #@profile
    def __call__(self, tensor_input,is_need_Orth_loss = False):
        dimension = len(tensor_input.size())
        tensor_input = tensor_input.to(self.device)
        size_input_batch = tensor_input.size(0)
        size_input_chanel = tensor_input.size(1)
        output_vector = tensor_input.clone().detach().view(size_input_batch, size_input_chanel, -1)
        cluster_ids_list= []
        cluster_centers_list = []
        Orth_loss = None
        # D == 2
        if dimension == 2:
            cluster_ids, cluster_centers = self.cluster(output_vector, debug=self.debug)
            cluster_ids_list.append(torch.from_numpy(cluster_ids).to(self.device))
            cluster_centers_list.append(cluster_centers)
        # D >= 3
        else:
            for batch in range(size_input_batch):
                y = output_vector.narrow(dim=0,start=batch, length=1).squeeze(0)
                cluster_ids, cluster_centers = self.cluster(y, debug=self.debug)
                cluster_ids_list.append(torch.from_numpy(cluster_ids).to(self.device))
                cluster_centers_list.append(cluster_centers)

        cluster_ids_stack = torch.stack(cluster_ids_list)
        cluster_centers_stack = torch.stack(cluster_centers_list)

        if is_need_Orth_loss:
            Orth_loss = self.Func_Orth(tensor_input, cluster_ids_stack)
            return Orth_loss, cluster_ids_stack, cluster_centers_stack
        else:
            return cluster_ids_stack, cluster_centers_stack


class KMeans_GPU_FAST():
    """
    A class for clustering 2-D tensors using K-Means algorithm.
    """
    def __init__(self,
                num_clusters,
                shift_threshold,
                max_iter,
                distance='euclidean',
                cluster_centers = [],
                device=torch.device('cuda')):

        self.num_clusters = num_clusters
        self.shift_threshold = shift_threshold
        self.max_iter = max_iter
        self.cluster_centers = cluster_centers
        self.device = device
        # self.initial_indices = None
        if distance == 'euclidean':
            self.pairwise_distance_function = pairwise_distance
        elif distance == 'cosine':
            self.pairwise_distance_function = pairwise_cosine
        else:
            raise NotImplementedError


    def initialize(self, X):
        num_samples = len(X)
        initial_indices = np.random.choice(num_samples, self.num_clusters, replace=False)
        initial_state = X[initial_indices]
        return initial_state

    #@profile
    def __call__(self, tensor_input, debug=False):
        if debug:
            time_start=time.time()

        X = tensor_input
        X = X.to(self.device)
        choice_points = np.ones(self.num_clusters)
        # init cluster center
        if type(self.cluster_centers) == list:
            initial_state = self.initialize(X)
        else:
            if debug:
                print('resuming cluster')
            initial_state = self.cluster_centers
            dis = self.pairwise_distance_function(X, initial_state, self.device)
            choice_points = torch.argmin(dis, dim=0)
            initial_state = X[choice_points]
            initial_state = initial_state.to(self.device)
        iteration = 0
        status = 0
        while status == 0:
            dis = self.pairwise_distance_function(X, initial_state, self.device).cpu().numpy()
            # choice_cluster = torch.argmin(dis, dim=1)
            choice_cluster = np.argmin(dis, axis=1)
            # save previous state
            initial_state_pre = initial_state.clone()

            for index in range(self.num_clusters):

                selected = np.where(choice_cluster == index)
                selected = X[selected]
                initial_state[index] = selected.mean(dim=0)

                dis_new = self.pairwise_distance_function(X, initial_state[index].unsqueeze(0), self.device).cpu().numpy()
                
                culuster_pos = np.argmin(dis_new, axis=0)
                while culuster_pos in choice_points[:index]:
                    dis_new[culuster_pos] = np.inf
                    culuster_pos = np.argmin(dis_new, axis=0)

                choice_points[index] = culuster_pos
            initial_state = X[choice_points]


            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
                ))

            iteration = iteration + 1

            if center_shift **2 < self.shift_threshold:
                status = 1
            if iteration >= self.max_iter:
                status = 2

            if debug:
                print("iter:{} center_shift:{:.5f}".format(iteration, center_shift))

        if debug:
            if status == 1:
                time_end=time.time()
                print('Time cost:{:.3f}'.format(time_end-time_start))
                print("Stoped for the center_shift!")
            else:
                time_end=time.time()
                print('Time cost:{:.3f}'.format(time_end-time_start))
                print("Stoped for the max_iter!")
        return choice_cluster, initial_state



def pairwise_distance(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1) # .squeeze()
    return dis

def pairwise_distance_numpy(data1, data2):
    # input type is numpy
    # data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    # A = data1.unsqueeze(dim=1)
    A = np.expand_dims(data1,axis=1)

    # 1*N*M
    # B = data2.unsqueeze(dim=0)
    B = np.expand_dims(data2,axis=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(axis=-1) # .squeeze()
    return dis




def pairwise_cosine(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1) #.squeeze()
    return cosine_dis
