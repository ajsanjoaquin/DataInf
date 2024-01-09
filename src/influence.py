from time import time
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle, os
from os.path import join
import torch

class IFEngine(object):
    def __init__(self):
        pass

    def preprocess_gradients(self, tr_grad_dict, val_grad_dict, noise_index=0):
        self.tr_grad_dict = tr_grad_dict
        self.val_grad_dict = val_grad_dict
        self.noise_index = noise_index

        self.n_train = len(self.tr_grad_dict.keys())
        self.n_val = len(self.val_grad_dict.keys())
        print("Computing avg gradient on validation dataset...")
        self.compute_val_grad_avg()

    def compute_val_grad_avg(self):
        # Compute the avg gradient on the validation dataset
        self.val_grad_avg_dict={}
        for weight_name in self.val_grad_dict[0]:
            self.val_grad_avg_dict[weight_name]=torch.zeros(self.val_grad_dict[0][weight_name].shape)
            for val_id in self.val_grad_dict:
                self.val_grad_avg_dict[weight_name] += self.val_grad_dict[val_id][weight_name] / self.n_val

    def compute_hvp_proposed(self, lambda_const_param=10):
        self.hvp_dict={}
        for weight_name in tqdm(self.val_grad_avg_dict):
            # lambda_const computation
            S=torch.zeros(len(self.tr_grad_dict.keys()))
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda
            
            # hvp computation
            hvp=torch.zeros(self.val_grad_avg_dict[weight_name].shape)
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                C_tmp = torch.sum(self.val_grad_avg_dict[weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                hvp += (self.val_grad_avg_dict[weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
            self.hvp_dict[weight_name] = hvp 

    def compute_IF(self):
        if_tmp_dict = {}
        for tr_id in tqdm(self.tr_grad_dict):
            if_tmp_value = 0
            for weight_name in self.val_grad_avg_dict:
                if_tmp_value += torch.sum(self.hvp_dict[weight_name]*self.tr_grad_dict[tr_id][weight_name])
            if_tmp_dict[tr_id]= -if_tmp_value 
            
        self.IF_arr = pd.Series(if_tmp_dict, dtype=float)

    def save_result(self, savedir, run_id=0):
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        self.IF_arr.to_csv(join(savedir, f"{run_id}.csv"))
class IFEngineGeneration(object):
    '''
    This class computes the influence function for every validation data point
    '''
    def __init__(self):
        pass
    def preprocess_gradients(self, tr_grad_dict, val_grad_dict):
        self.tr_grad_dict = tr_grad_dict
        self.val_grad_dict = val_grad_dict

        self.n_train = len(self.tr_grad_dict.keys())
        self.n_val = len(self.val_grad_dict.keys())

    def compute_hvp_proposed(self, lambda_const_param=10):
        hvp_proposed_dict=defaultdict(dict)
        print("Computing for each validation point")
        for val_id in tqdm(self.val_grad_dict.keys()):
            for weight_name in self.val_grad_dict[val_id]:
                # lambda_const computation
                S=torch.zeros(len(self.tr_grad_dict.keys()))
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                    S[tr_id]=torch.mean(tmp_grad**2)
                lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

                # hvp computation
                hvp=torch.zeros(self.val_grad_dict[val_id][weight_name].shape)
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                    C_tmp = torch.sum(self.val_grad_dict[val_id][weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                    hvp += (self.val_grad_dict[val_id][weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
                hvp_proposed_dict[val_id][weight_name] = hvp
        self.hvp_dict = hvp_proposed_dict

    def compute_IF(self):
        if_tmp_dict = defaultdict(dict)
        for tr_id in self.tr_grad_dict:
            for val_id in self.val_grad_dict:
                if_tmp_value = 0
                for weight_name in self.val_grad_dict[0]:
                    if_tmp_value += torch.sum(self.hvp_dict[val_id][weight_name]*self.tr_grad_dict[tr_id][weight_name])
                if_tmp_dict[tr_id][val_id]=if_tmp_value

        self.IF_dict = pd.DataFrame(if_tmp_dict, dtype=float)   

    def save_result(self, savedir, run_id=0):
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        self.IF_dict.to_csv(join(savedir, f"{run_id}.csv"), index=False)