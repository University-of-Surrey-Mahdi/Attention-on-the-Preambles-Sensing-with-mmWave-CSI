### Post process to output localization result

import numpy as np
import torch
from itertools import combinations_with_replacement

# Fixed parameters
DELTA       = 1e-6
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Variable parameters, but set the default
NUM_ALL     = 8 # 1
NUM_NEU     = 9 # 8
ALL_CAND    = []

# Set variable parameters
def set_param(num_all, num_neu):
    global NUM_ALL
    global NUM_NEU
    global ALL_CAND
    
    NUM_ALL = num_all
    NUM_NEU = num_neu
    
    ALL_CAND = []
    for count_i in range(NUM_ALL):
        list_cand   = list((combinations_with_replacement(list(range(NUM_NEU)),(count_i+1))))
        arr_cand    = np.zeros((len(list_cand),NUM_NEU))
        for cand_i in range(len(list_cand)):
            for count_i_i in range(count_i+1):
                arr_cand[cand_i,list_cand[cand_i][count_i_i]] += 1/(count_i+1)
        
        arr_cand[arr_cand==0.0] = DELTA
        ALL_CAND.append(torch.from_numpy(arr_cand.astype(np.float32)).to(device))

# Post process
def pred(pred_y, t_c_local):
    
    pred_result     = torch.from_numpy(np.zeros((len(pred_y), NUM_NEU)).astype(np.float32))
    
    for i in range(len(pred_y)):
        pred_y_local = pred_y[i, :]
        count_idx = int(t_c_local[i] - 1)
        cand = ALL_CAND[count_idx]
        
        best_idx = torch.argmin(torch.sum(- pred_y_local * torch.log(cand), 1))
        ans_raw = torch.clone(cand[best_idx])
        ans_raw[ans_raw==DELTA] = 0.0
        ans     = ans_raw * (count_idx+1)
        
        pred_result[i, :] = torch.clone(ans)
    
    return pred_result.to(device)
