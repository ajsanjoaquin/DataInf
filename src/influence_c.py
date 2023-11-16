from time import time
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle
import os
from os.path import join
import torch

class IFEngineGeneration(object):
    '''
    This class computes the influence function for every validation data point
    '''
    def __init__(self, tr_chunks, savedir, train_len):
        self.time_dict = defaultdict(list)
        self.hvp_dict = defaultdict(list)
        self.IF_dict = defaultdict(list)
        self.tr_chunks = [i+1 for i in range(tr_chunks)]
        # chunks path
        self.savedir = savedir

        self.val_grad_dict = load_pkl(join(self.savedir, 'val.pkl'))
        self.n_train = train_len
        self.n_val = len(self.val_grad_dict.keys())

    def compute_hvp_proposed(self, lambda_const_param=10):
        start_time = time()
        hvp_proposed_dict=defaultdict(dict)

        for val_id in tqdm(range(self.n_val)):
            print("Looping through model weights...")
            for weight_name in tqdm(self.val_grad_dict[val_id]):
                # lambda_const computation
                S=torch.zeros(self.n_train)

                for c in self.tr_chunks:
                    tr_grad_dict = load_pkl(join(self.savedir, 'tr_%i.pkl' % c))
                    for tr_id in tr_grad_dict:
                        tmp_grad = tr_grad_dict[tr_id][weight_name]
                        S[tr_id]=torch.mean(tmp_grad**2)
                lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

                # hvp computation
                hvp=torch.zeros(self.val_grad_dict[val_id][weight_name].shape)
                for c in self.tr_chunks:
                    tr_grad_dict = load_pkl(join(self.savedir, 'tr_%i.pkl' % c))
                    for tr_id in tr_grad_dict:
                        tmp_grad = tr_grad_dict[tr_id][weight_name]
                        C_tmp = torch.sum(self.val_grad_dict[val_id][weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                        hvp += (self.val_grad_dict[val_id][weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
                hvp_proposed_dict[val_id][weight_name] = hvp

            print("Saving intermediate results...")
            with open(join(self.savedir, f"hvps/hvp_val_{val_id}.pkl"),'wb') as file:
                pickle.dump(hvp_proposed_dict, file, pickle.HIGHEST_PROTOCOL)

        self.hvp_dict['proposed'] = hvp_proposed_dict
        self.time_dict['proposed'] = time()-start_time

    def compute_IF(self):
        for method_name in self.hvp_dict:
            print("Computing IF for method: ", method_name)
            if_tmp_dict = defaultdict(dict)
            for c in self.tr_chunks:
                tr_grad_dict = load_pkl(join(self.savedir, 'tr_%i.pkl' % c))
                for tr_id in tr_grad_dict:
                    for val_id in self.val_grad_dict:
                        if_tmp_value = 0
                        for weight_name in self.val_grad_dict[0]:
                            if_tmp_value += torch.sum(self.hvp_dict[method_name][val_id][weight_name]*tr_grad_dict[tr_id][weight_name])
                        if_tmp_dict[tr_id][val_id]=if_tmp_value

            self.IF_dict[method_name] = pd.DataFrame(if_tmp_dict, dtype=float)   

    def save_result(self, savedir, run_id=0):
        results={}
        results['runtime']=self.time_dict
        results['influence']=self.IF_dict

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        with open(join(savedir, f"results_{run_id}.pkl"),'wb') as file:
            pickle.dump(results, file)
def load_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data