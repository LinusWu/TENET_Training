import numpy as np
import torch
import sys
sys.path.append('./CFG/')
sys.path.append('./GMW/')
import CFG.group as Group
import CFG.Orthogonal_loss as Orthogo


class Tenet:
    def __init__(self, model, group_numbers, feature_id = 3, total_layer = 4):
        self.model = model
        self.feature_id = feature_id
        self.grad_collector = Gradient_Collector(self.model)
        self.group_numbers = group_numbers
        self.CFG = Group.Cluster_GPU_FAST(self.group_numbers)
        self.total_layer = total_layer
        self.loss_Orthogo = Orthogo.OrthogonalLoss(self.group_numbers)


    def gradient_collect(self, x, target):
        features, grads = self.grad_collector(x, target)
        return features, grads


    def _Reverse_func(self,z):
        return 1/(1 + np.exp(z))


    def GMW(self, x, features, grads, is_need_orthogonal=True):
        # group and get mask
        mask_tensor = self.get_masktensor(features, grads).cuda()
        # inference 
        self.model.train()
        self.model.zero_grad()
        for i in range(self.total_layer):
            x = self.model.feature_extract(x, layer_num = i+1)
            if i+1 == self.feature_id:
                if is_need_orthogonal:
                    self.Orth_loss = self.loss_Orthogo(x,self.cluster_index)
                else:
                    self.Orth_loss = 0
                x = torch.mul(x, mask_tensor)

        # forward using weighted feature maps
        x = self.model.forward_classifier(x)
        return x, self.Orth_loss


    def get_masktensor(self, features, grads):
        selected_feature = features[0]
        selected_grad = grads[0]
        b,c,h,w = selected_feature.size()
        self.cluster_index, self.cluster_centers = self.CFG(selected_feature)
        cluster_index = self.cluster_index.cpu().numpy()
        group_mask = np.zeros((b,self.group_numbers,h,w))
        group_weight = np.zeros((b, self.group_numbers))
        mask_tensor = np.zeros((b,c,h,w),dtype='float32')

        self.feature = selected_feature.detach().cpu().numpy()
        self.w_dict = selected_grad

        for i in range(b):
            # batch-index 
            for group_id in range(self.group_numbers):
                selected = np.where(cluster_index[i] == group_id)
                W = self.w_dict[i][selected].mean(axis=(1,2))
                group_weight[i][group_id] = W.sum()
                # group_weight[i][group_id] = W.mean()

                if group_weight[i][group_id] <= 0:
                    group_mask[i][group_id] = 1.0

                else:
                    F = self.feature[i][selected].transpose(1,2,0)
                    m = F.dot(W)
                    # normalization
                    m = (m - np.mean(m)) / ((np.std(m))+1e-8)
                    m = self._Reverse_func(m)
                    group_mask[i][group_id] = m.copy()

                mask_tensor[i][selected] = group_mask[i][group_id]
        self.mask_tensor = torch.from_numpy(mask_tensor)
        return self.mask_tensor

    def __call__(self, x, target):
        # get features and gradient
        features, grads = self.gradient_collect(x, target)
        out, loss_orth = self.GMW(x,features,grads)
        return  out, loss_orth


# rethink
class Gradient_Collector():
    def __init__(self, model):
        self.model = model
        self.extractor = ModelWapper(self.model)
        self.channel_weighting_res={}
        self.feature_dict = {}
        self.sig = torch.nn.Sigmoid()

    def __call__(self, x, target):
        self.model.eval()
        features, output = self.extractor(x)
        output = self.sig(output)
        one_hot = np.zeros((output.size()[0], output.size()[-1]), dtype=np.float32)
        index = target.cpu().numpy()
        for (b,j) in enumerate(index):
            one_hot[b,j] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).cuda()
        L_d = torch.sum(one_hot * output)
        # first backward for getting gradients
        self.model.zero_grad()
        L_d.backward(retain_graph=True)
        grads = self.extractor.get_gradients()
        return features, grads


class FeatureExtractor():
    def __init__(self, model):
        self.model = model
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad.clone().detach().cpu().numpy())

    def __call__(self, x, feature_id = 3, layer_total=4):
        outputs = []
        self.gradients = []
        for i in range(layer_total):
            x = self.model.feature_extract(x,layer_num = i+1)
            # regist host function for saving gradient
            if feature_id == i + 1:
                x.register_hook(self.save_gradient)
                outputs.insert(0, x)
        return outputs, x

class ModelWapper():
    def __init__(self, model):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, x = self.feature_extractor(x)
        x = self.model.forward_classifier(x)
        return target_activations, x

