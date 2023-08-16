
# import required modules to clean cache
import torch
import gc
# clean cache
gc.collect()
torch.cuda.empty_cache()   
# define device type
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# import required modules to load ground truth
import numpy as np
import sub_modules.load_gt as load_gt
# import required modules to pre-process
from scipy.io import loadmat
import sub_modules.pre_chest as pre_chest
import sub_modules.pre_doppler as pre_doppler
import sub_modules.pre_angle as pre_angle
# import ML module
import sub_modules.ml_models as ml_models
import sub_modules.previousDNN as previousDNN
import sub_modules.previousGEO as previousGEO
import copy
import tensorflow as tf
# import required modules to output prediction result
import sub_modules.post as post
# import required modules to report
import sub_modules.report as report
# import required modules to plot
import matplotlib.pyplot as plt

# dataset path
DATA_PATH   = "E:/WALDO/PS-002-WALDO-main/dataset/"     # Parent path
TRAIN_GT    = DATA_PATH + "mlOutput/groundTruth_correct.txt"            # Ground truth of training dataset
VALID_GT    = DATA_PATH + "mlOutput/groundTruthValidation_correct.txt"  # Ground truth of validation dataset
TRAIN_RS    = DATA_PATH + "rxSignal/"                   # Received signals of training dataset
VALID_RS    = DATA_PATH + "validation/rxSignal/"        # Received signals of validation dataset
COV_RS      = DATA_PATH + "cov/"                        # Covariance matrices of received signals for LMMSE

# RUN MODE (Default: All True. If you set some modes as False, the process will be skipped.)
PREPRO_MODE         = True          # If True, pre-process will run. Else, pre-processed data will be loaded.
PREPRO_SAVE_MODE    = True          # If True AND PREPRO_MODE == True, pre-processed data will be saved. After that, you can set PREPRO_MODE as False (skip pre-process and load the pre-processed data)
FULL_DATA_MODE      = True          # If False, program runs with small dataset and small epoch. It is worthy if you want to check if program runs without ERROR.
TRAIN_MODE          = True          # If False, no learning runs, so it just installed pre-learned model.
SOLUTION_MODE       = {"ViT": True, "CNN": True, "DNN": True, "GEO": True}  # if True, Program runs for the solution

# Fixed parameters
if FULL_DATA_MODE:
    N_TRAIN = 15578  # Number of data in training dataset
    N_VAL = 1858  # Number of data in validation dataset
    EPOCH_FULL = 200  # Number of epoch when learning
    EPOCH_CURR = 40  # Number of epoch when curriculum learning
    SNR_LIST = [-18, 0, 18]  # List of target SNR
else:
    N_TRAIN = 100  # Number of data in training dataset
    N_VAL = 50  # Number of data in validation dataset
    EPOCH_FULL = 10  # Number of epoch when learning
    EPOCH_CURR = 10  # Number of epoch when curriculum learning
    SNR_LIST = [-18, 0, 18]  # List of target SNR
N_L             = 44        # Tap length to represent path delay
N_RX            = 4         # Number of antennas at RX device (4)
N_TX            = 4         # Number of antennas at TX device (4)
N_CH            = 128       # Number of snap shots (128)
THETA_R         = 0.2       # Phase shift at angular domain analysis (tuned parameter is 0.2)
THETA_T         = 0.8       # Phase shift at angular domain analysis (tuned parameter is 0.8)
MAX_PERSONS     = 8         # Max number of persons in a room (8)
MAX_SECTOR      = 9         # Number of sectors in a room (9)
# Hyper-parameters for each solution ("None" means it is not used)
# K_C/L:    K values used in doppler domain analysis for counting/localization
# SR_C/L:   If True, CSI corresponding to static objects are removed for counting/localization
# B:        Batch size
# Lr_C:     Learning rate in training ML for counting
# Lr_L_18:  Learning rate in training ML for localization at first step
# Lr_L_Cur: Learning rate in training ML for localization at successive curriculum step
# WD:       Weight decay
# MT:       Model type (name)
# PP:       Pre-Process mode (LSE oe LMMSE)
HP_ViT          = {"K_C": 1,  "K_L": 1,     "SR_C": True,  "SR_L": True,  "B": 16,   "Lr_C": 3e-4, "Lr_L_18": 6e-4, "Lr_L_Cur": 3e-4, "WD": 0.0,  "MT": "ViT",            "PP": "LSE"}  # proposed ViT solution
HP_CNN          = {"K_C": 16, "K_L": 8,     "SR_C": False, "SR_L": True,  "B": 32,   "Lr_C": 5e-6, "Lr_L_18": 1e-5, "Lr_L_Cur": 5e-6, "WD": 0.01, "MT": "CNN",            "PP": "LSE"}  # proposed CNN solution
HP_DNN          = {"K_C": 8,  "K_L": 8,     "SR_C": False, "SR_L": False, "B": None, "Lr_C": None, "Lr_L_18": None, "Lr_L_Cur": None, "WD": None, "MT": "previousDNN",    "PP": "LSE"}  # previous DNN solution
HP_GEO          = {"K_C": 2,  "K_L": None,  "SR_C": False, "SR_L": None,  "B": None, "Lr_C": None, "Lr_L_18": None, "Lr_L_Cur": None, "WD": None, "MT": "previousGEO",    "PP": "LSE"}  # previous geometry-based solution


### Pre-process function
# Pre-process data
def PreProcess(path, NumTestSamples, snr_val_list, mean_k, pp_mode, theta_r, theta_t):
    
    # Matrix which pre-processed data will be stored 
    h_rad = np.zeros((NumTestSamples * int(len(SNR_LIST)), N_RX, N_TX, (N_L + 1), int(N_CH/mean_k)), dtype=float)
            
    # loop for all SNR
    for snr_idx, snr_val in enumerate(snr_val_list):
        # loop for all data
        for i in range(NumTestSamples):
            
            # read received signal (.mat)
            rx = loadmat(path + "snr" + str(snr_val) + '/rxSigCh' + str(i) + '.mat')['rxNoisy'] # [received signal, N_RX, N_CH] 
            # CH estimation (Pre-process 1)            
            if pp_mode == "LSE":
                H       = pre_chest.LSE(rx)                 # [N_RX, N_TX, (N_L + 1), N_CH]   (complex value)
            elif pp_mode == "LMMSE":
                H       = pre_chest.LMMSE(COV_RS, rx)       # [N_RX, N_TX, (N_L + 1), N_CH]   (complex value)
            # Doppler domain analysis (Pre-process 2)
            H_VEL   = pre_doppler.process(H, mean_k)        # [N_RX, N_TX, (N_L + 1), N_CH/K] (complex value)
            # Angular domain analysis (Pre-process 3)
            pre_angle.set_phase_shift(theta_r, theta_t)     # set the phase shift
            H_VEL_ANG = pre_angle.process(H_VEL, mean_k)    # [N_RX, N_TX, (N_L + 1), N_CH/K] (complex value)
            H_VEL_ANG = abs(H_VEL_ANG)                      # [N_RX, N_TX, (N_L + 1), N_CH/K] (real value)

            if PREPRO_SAVE_MODE:
                # It is required to save pre-processed CSI
                if "validation" in path:
                    path_tv = "val/"
                else:
                    path_tv = "train/"
                if theta_r == 0.0 and theta_t == 0.0:
                    path_sh = "_all/"
                else:
                    path_sh = "_all_shift/"
                with open(DATA_PATH + "Hf_vel" + str(mean_k) + path_sh + path_tv + "snr" + str(snr_val) + "/hfv" + str(i) + ".npy", "wb") as fw:
                    np.save(fw, H_VEL_ANG)

            
            # Store pre-processed data
            h_rad[snr_idx * NumTestSamples + i, :, :, :, :] = H_VEL_ANG
    return h_rad

# Load data
def LoadPreProcess(mean_k, pp_mode, theta_r, theta_t, remove):
    
    if PREPRO_MODE:
        # Calculate covariance for LMMSE
        if pp_mode == "LMMSE":
            _, _, _, _, _, _, _ = pre_chest.CalcCOV(TRAIN_RS, COV_RS, int(SNR_LIST[-1]), N_TRAIN)
        # Pre-process data
        h_rad_train = PreProcess(TRAIN_RS, N_TRAIN, SNR_LIST, mean_k, pp_mode, theta_r, theta_t)
        h_rad_val   = PreProcess(VALID_RS, N_VAL, SNR_LIST, mean_k, pp_mode, theta_r, theta_t)
    else:
        # Load pre-processed data
        h_rad_train = np.zeros((N_TRAIN * int(len(SNR_LIST)), N_RX, N_TX, (N_L + 1), int(N_CH/mean_k)), dtype=float)
        h_rad_val   = np.zeros((N_VAL * int(len(SNR_LIST)), N_RX, N_TX, (N_L + 1), int(N_CH/mean_k)), dtype=float)
        
        if theta_r == 0.0 and theta_t == 0.0:
            path_sh = "_all/"
        else:
            if pp_mode == "LSE":
                path_sh = "_all_shift/"
            elif pp_mode == "LMMSE":
                path_sh = "_all_shift_lmmse/"
        for snr_idx, snr_val in enumerate(SNR_LIST):
            for i in range(N_TRAIN):
                with open(DATA_PATH + "Hf_vel" + str(mean_k) + path_sh + "train/snr" + str(snr_val) + "/hfv" + str(i) + ".npy", "rb") as f:
                    h_rad_train[i + snr_idx * N_TRAIN,:,:,:,:] = np.load(f)
            for i in range(N_VAL):
                with open(DATA_PATH + "Hf_vel" + str(mean_k) + path_sh + "val/snr" + str(snr_val) + "/hfv" + str(i) + ".npy", "rb") as f:
                    h_rad_val[i + snr_idx * N_VAL,:,:,:,:] = np.load(f)
        
    # remove the CSI corresponding static objects (CH == 0) by replacing values with 0.0
    if remove:
        h_rad_train[:,:,:,:,0]  = 0.0
        h_rad_val[:,:,:,:,0]    = 0.0
    
    return h_rad_train, h_rad_val



### Function for training ML for counting
def ML_Counting_Training(mean_k, model_type, model_count_name, h_rad_train, count_array_train, h_rad_val, count_array_val, lr, wd, bs):

    # Loading ML model for counting
    model_count         = ml_models.init_NN(mean_k, 1, MAX_PERSONS, model_type)
    # Train ML for counting
    if TRAIN_MODE:
        # train the model
        model_count, train_history_c, trainacc_history_c, valid_history_c, validacc_history_c \
            = ml_models.train_NN(model_count, h_rad_train, count_array_train, count_array_train, h_rad_val,
                                 count_array_val, count_array_val, EPOCH_FULL, lr, wd, bs)
        # save the model
        model_count.eval()
        torch.save(model_count.state_dict(), model_count_name + ".pth")

    else:
        # load the model
        model_count.load_state_dict(torch.load(model_count_name + ".pth", map_location=torch.device(device)))
        train_history_c, trainacc_history_c, valid_history_c, validacc_history_c = [], [], [], []
    
    return model_count, train_history_c, trainacc_history_c, valid_history_c, validacc_history_c
    


###  Function for training previous DNN for counting
def ML_Counting_previousDNN_Training(mean_k, model_count_name, h_rad_train, count_array_train):
    
    # Loading ML model for counting
    model_count = previousDNN.CountingPeople(mean_k)
    # Train ML for counting
    if TRAIN_MODE:
        # train the model
        model_count.fit([h_rad_train], count_array_train, batch_size=32, epochs=200, verbose=0)
        # save the model
        model_count.save(model_count_name + ".keras")

    else:
        # load the model
        model_count = tf.keras.models.load_model(model_count_name + ".keras", compile=False)
    
    return model_count



### Function for testing ML for counting
def ML_Counting_Testing(model_count, h_rad_train, h_rad_val, count_array_val, bs):
    
    post.set_param(1, MAX_PERSONS)
    pred_c_train    = []
    pred_c          = []
    for snr_idx, snr_val in enumerate(SNR_LIST):
        # Prediction on test data
        Y_pred_c_train      = ml_models.predict_NN(model_count, h_rad_train[N_TRAIN*snr_idx:N_TRAIN*(snr_idx+1),:,:,:,:], bs)
        Y_pred_c            = ml_models.predict_NN(model_count, h_rad_val[N_VAL*snr_idx:N_VAL*(snr_idx+1),:,:,:,:], bs)
        
        pred_tr_c_train     = post.pred((torch.from_numpy(Y_pred_c_train)).to(device), (torch.from_numpy(count_array_train.sum(axis=1))).to(device))
        pred_tr_c           = post.pred((torch.from_numpy(Y_pred_c)).to(device), (torch.from_numpy(count_array_val.sum(axis=1))).to(device))
        
        pred_c_train.append( np.argmax(pred_tr_c_train.detach().cpu().numpy(), axis=1) + 1 )
        pred_c.append( np.argmax(pred_tr_c.detach().cpu().numpy(), axis=1) + 1 )
    
    return pred_c_train, pred_c



### Function for curriculum training ML for localization
def ML_Localization_Training(mean_k, model_type, model_localization_name_part, h_rad_train, count_array_train, location_array_ev_train, h_rad_val, count_array_val, location_array_ev_val, lr0, lrc, wd, bs, model_count=0):
    SNR_LIST_rev = list(reversed(SNR_LIST))
    model_localization_dict = {}
    for snr_val_train in SNR_LIST_rev:
        model_localization_dict[snr_val_train] = []
    
    train_history_c, trainacc_history_c, valid_history_c, validacc_history_c = list(range(len(SNR_LIST))), list(range(len(SNR_LIST))), list(range(len(SNR_LIST))), list(range(len(SNR_LIST)))
    # Train ML for localization
    # Curriculum training
    for snr_idx_train, snr_val_train in zip([2, 1, 0], SNR_LIST_rev):
        
        model_localization      = ml_models.init_NN(mean_k, MAX_PERSONS, MAX_SECTOR, model_type)
        model_localization_name = model_localization_name_part + "_" + str(snr_val_train) + "dB.pth"
        model_localization_high_name = model_localization_name_part + "_" + str(snr_val_train + int(SNR_LIST[1] - SNR_LIST[0])) + "dB.pth"
        
        if TRAIN_MODE:
            # TRANSFER and TRAIN model
            if snr_val_train == int(SNR_LIST[-1]):
                # transfer from counting ML if any
                if model_count == 0:
                    pass
                else:
                    model_tmp = copy.deepcopy(model_count)
                    model_tmp.mlp_head[1] = torch.nn.Linear(model_localization.mlp_head[1].in_features,
                                                            model_localization.mlp_head[1].out_features).to(device)
                    model_localization = model_tmp

                # train the model
                model_localization, train_history_c[snr_idx_train], trainacc_history_c[snr_idx_train], valid_history_c[
                    snr_idx_train], validacc_history_c[snr_idx_train] \
                    = ml_models.train_NN(model_localization, h_rad_train, count_array_train, location_array_ev_train,
                                         h_rad_val, count_array_val, location_array_ev_val, EPOCH_FULL, lr0, wd, bs)

            else:
                # transfer from ML of higher SNR
                model_localization.load_state_dict(
                    torch.load(model_localization_high_name, map_location=torch.device(device)))

                # train the model
                model_localization, train_history_c[snr_idx_train], trainacc_history_c[snr_idx_train], valid_history_c[
                    snr_idx_train], validacc_history_c[snr_idx_train] \
                    = ml_models.train_NN(model_localization, h_rad_train, count_array_train, location_array_ev_train,
                                         h_rad_val, count_array_val, location_array_ev_val, EPOCH_CURR, lrc, wd, bs)

            # save the model
            model_localization.eval()
            torch.save(model_localization.state_dict(), model_localization_name)

        else:
            # load the model
            model_localization.load_state_dict(torch.load(model_localization_name, map_location=torch.device(device)))
            train_history_c, trainacc_history_c, valid_history_c, validacc_history_c = [], [], [], []
            
        model_localization_dict[snr_val_train] = model_localization
    
    return model_localization_dict, train_history_c, trainacc_history_c, valid_history_c, validacc_history_c
       


###  Function for training previous DNN for counting
def ML_Localization_previousDNN_Training(mean_k, model_localization_name, h_rad_train, location_array_train):
    
    # Loading ML model for localization
    model = previousDNN.load_NNweights(mean_k)
    # Train ML for localization
    if TRAIN_MODE:
        # train the model
        model.fit([h_rad_train], location_array_train, batch_size=32, epochs=200, verbose=0)
        # save the model
        model.save(model_localization_name + ".keras")
    
    else:
        # load the model
        model = tf.keras.models.load_model(model_localization_name + ".keras", compile=False)
        
    return model



### Function for testing ML for localization
def ML_Localization_Testing(model_localization_dict, h_rad_train, h_rad_val, pred_c_train, pred_c, bs):
    
    post.set_param(MAX_PERSONS, MAX_SECTOR)
    
    pred_l_train    = []
    pred_l          = []
    for snr_idx, snr_val in enumerate(SNR_LIST):
        # Prediction on test data
        Y_pred_l_train  = ml_models.predict_NN(model_localization_dict[snr_val], h_rad_train[N_TRAIN*snr_idx:N_TRAIN*(snr_idx+1),:,:,:,:], bs)
        Y_pred_l        = ml_models.predict_NN(model_localization_dict[snr_val], h_rad_val[N_VAL*snr_idx:N_VAL*(snr_idx+1),:,:,:,:], bs)
        
        pred_tr_l_train = post.pred((torch.from_numpy(Y_pred_l_train)).to(device), pred_c_train[snr_idx])
        pred_tr_l       = post.pred((torch.from_numpy(Y_pred_l)).to(device), pred_c[snr_idx])
        
        pred_l_train.append( pred_tr_l_train.detach().cpu().numpy() )
        pred_l.append( pred_tr_l.detach().cpu().numpy() )
    
    return pred_l_train, pred_l


### Function for report accuracy
def Report(count_array_train, location_array_ev_train, count_array_val, location_array_ev_val, pred_c_train, pred_l_train, pred_c, pred_l, model_type):

    # Print title
    print("##########  " + "{:30}".format(model_type + " Accuracy[%]") + "   ##########")
    # Evaluate the accuracy
    c_train_w, l_train_w, c_val_w, l_val_w = report.evaluate(count_array_train, location_array_ev_train, count_array_val, location_array_ev_val, pred_c_train, pred_l_train, pred_c, pred_l)
    # Save the result to a file (.txt)
    report.save_sector_form(pred_l, SNR_LIST, "mlResult" + model_type)
    # Print the result to console
    report.print_result(c_train_w, l_train_w, c_val_w, l_val_w, SNR_LIST)

    return c_train_w, l_train_w, c_val_w, l_val_w



##### MAIN FUNCTION START #####

### proposed ViT
if SOLUTION_MODE["ViT"]:
    HP = HP_ViT.copy()
    # Load ground truth values
    count_array_train, location_array_train, location_array_ev_train = load_gt.load(TRAIN_GT, N_TRAIN, 10)
    count_array_val, location_array_val, location_array_ev_val = load_gt.load(VALID_GT, N_VAL, 10)
    # Load and pre-process received signals for counting
    h_rad_train, h_rad_val  = LoadPreProcess(HP["K_C"], HP["PP"], THETA_R, THETA_T, HP["SR_C"])
    # Training ML for counting
    model_count, _, _, _, _ = ML_Counting_Training(HP["K_C"], HP["MT"], "model/model_count_" + HP["MT"] + "_" + HP["PP"], h_rad_train, count_array_train, h_rad_val, count_array_val, HP["Lr_C"], HP["WD"], HP["B"])
    # Testing ML for counting
    pred_c_train, pred_c    = ML_Counting_Testing(model_count, h_rad_train, h_rad_val, count_array_val, HP["B"])
    # Load and pre-process received signals for localization
    if HP["K_C"] != HP["K_L"] or HP["SR_C"] != HP["SR_L"]:
        h_rad_train, h_rad_val  = LoadPreProcess(HP["K_L"], HP["PP"], THETA_R, THETA_T, HP["SR_L"])
    # Training ML for localization
    model_localization_dict, _, _, _, _ = ML_Localization_Training(HP["K_L"], HP["MT"], "model/model_localization_" + HP["MT"] + "_" + HP["PP"], h_rad_train, count_array_train, location_array_ev_train, h_rad_val, count_array_val, location_array_ev_val, HP["Lr_L_18"], HP["Lr_L_Cur"], HP["WD"], HP["B"], model_count)
    # Testing ML for localization
    pred_l_train, pred_l    = ML_Localization_Testing(model_localization_dict, h_rad_train, h_rad_val, pred_c_train, pred_c, HP["B"])
    # Report accuracy
    c_train_ViT, l_train_ViT, c_val_ViT, l_val_ViT = Report(count_array_train, location_array_ev_train, count_array_val, location_array_ev_val, pred_c_train, pred_l_train, pred_c, pred_l, HP["MT"])

### proposed CNN
if SOLUTION_MODE["CNN"]:
    HP = HP_CNN.copy()
    # Load ground truth values
    count_array_train, location_array_train, location_array_ev_train = load_gt.load(TRAIN_GT, N_TRAIN, 10)
    count_array_val, location_array_val, location_array_ev_val = load_gt.load(VALID_GT, N_VAL, 10)
    # Load and pre-process received signals for counting
    h_rad_train, h_rad_val  = LoadPreProcess(HP["K_C"], HP["PP"], THETA_R, THETA_T, HP["SR_C"])
    # Training ML for counting
    model_count, _, _, _, _ = ML_Counting_Training(HP["K_C"], HP["MT"], "model/model_count_" + HP["MT"] + "_" + HP["PP"], h_rad_train, count_array_train, h_rad_val, count_array_val, HP["Lr_C"], HP["WD"], HP["B"])
    # Testing ML for counting
    pred_c_train, pred_c    = ML_Counting_Testing(model_count, h_rad_train, h_rad_val, count_array_val, HP["B"])
    # Load and pre-process received signals for localization
    if HP["K_C"] != HP["K_L"] or HP["SR_C"] != HP["SR_L"]:
        h_rad_train, h_rad_val  = LoadPreProcess(HP["K_L"], HP["PP"], THETA_R, THETA_T, HP["SR_L"])
    # Training ML for localization
    model_localization_dict, _, _, _, _ = ML_Localization_Training(HP["K_L"], HP["MT"], "model/model_localization_" + HP["MT"] + "_" + HP["PP"], h_rad_train, count_array_train, location_array_ev_train, h_rad_val, count_array_val, location_array_ev_val, HP["Lr_L_18"], HP["Lr_L_Cur"], HP["WD"], HP["B"], 0)
    # Testing ML for localization
    pred_l_train, pred_l    = ML_Localization_Testing(model_localization_dict, h_rad_train, h_rad_val, pred_c_train, pred_c, HP["B"])
    # Report accuracy
    c_train_CNN, l_train_CNN, c_val_CNN, l_val_CNN = Report(count_array_train, location_array_ev_train, count_array_val, location_array_ev_val, pred_c_train, pred_l_train, pred_c, pred_l, HP["MT"])

### previous DNN-based solution
if SOLUTION_MODE["DNN"]:
    HP = HP_DNN.copy()
    # Load ground truth values
    count_array_train, location_array_train, location_array_ev_train = load_gt.load(TRAIN_GT, N_TRAIN, 3)
    count_array_val, location_array_val, location_array_ev_val = load_gt.load(VALID_GT, N_VAL, 3)
    # Load and pre-process received signals for counting
    h_rad_train, h_rad_val  = LoadPreProcess(HP["K_C"], HP["PP"], 0.0, 0.0, HP["SR_C"])
    # Training ML for counting
    model_count = ML_Counting_previousDNN_Training(HP["K_C"], "model/model_count_" + HP["MT"], h_rad_train, count_array_train)
    # Load and pre-process received signals for localization
    if HP["K_C"] != HP["K_L"] or HP["SR_C"] != HP["SR_L"]:
        h_rad_train, h_rad_val  = LoadPreProcess(HP["K_L"], HP["PP"], 0.0, 0.0, HP["SR_L"])
    # Training ML for localization
    model_localization = ML_Localization_previousDNN_Training(HP["K_L"], "model/model_localization_" + HP["MT"], h_rad_train, location_array_train)
    # Testing ML for counting/localization
    pred_c_train    = []
    pred_c          = []
    pred_l_train    = []
    pred_l          = []
    for snr_idx, snr_val in enumerate(SNR_LIST):
        # Prediction on test data
        Y_pred_train = model_localization.predict([h_rad_train[N_TRAIN*snr_idx:N_TRAIN*(snr_idx+1), :, :, :, :]], verbose=0)
        Y_pred = model_localization.predict([h_rad_val[N_VAL*snr_idx:N_VAL*(snr_idx+1), :, :, :, :]], verbose=0)
        # Conversion of predictions to strings in sector format
        Y_pred_sector_train = previousDNN.Testing(Y_pred_train, 4)
        Y_pred_sector = previousDNN.Testing(Y_pred, 4)
        len_pred_tr_train, Y_pred_tr_train, lin_tr_train = previousDNN.sector_form(h_rad_train[N_TRAIN*snr_idx:N_TRAIN*(snr_idx+1), :, :, :, :], Y_pred_train, Y_pred_sector_train, model_count.predict([h_rad_train[N_TRAIN*snr_idx:N_TRAIN*(snr_idx+1), :, :, :, :]], verbose=0))
        len_pred_tr, Y_pred_tr, lin_tr = previousDNN.sector_form(h_rad_val[N_VAL*snr_idx:N_VAL*(snr_idx+1), :, :, :, :], Y_pred, Y_pred_sector, model_count.predict([h_rad_val[N_VAL*snr_idx:N_VAL*(snr_idx+1), :, :, :, :]], verbose=0))
        
        pred_c_train.append( model_count.predict([h_rad_train[N_TRAIN*snr_idx:N_TRAIN*(snr_idx+1), :, :, :, :]], verbose=0) )
        pred_c.append( model_count.predict([h_rad_val[N_VAL*snr_idx:N_VAL*(snr_idx+1), :, :, :, :]], verbose=0) )
        pred_l_train.append( Y_pred_tr_train)
        pred_l.append( Y_pred_tr)
    # Report accuracy
    c_train_DNN, l_train_DNN, c_val_DNN, l_val_DNN = Report(count_array_train, location_array_ev_train, count_array_val, location_array_ev_val, (np.argmax(pred_c_train, axis=2) + 1), pred_l_train, (np.argmax(pred_c, axis=2) + 1), pred_l, HP["MT"])

### previous Geometry-based solution
if SOLUTION_MODE["GEO"]:
    HP = HP_GEO.copy()
    # Load ground truth values
    count_array_train, location_array_train, location_array_ev_train = load_gt.load(TRAIN_GT, N_TRAIN, 3)
    count_array_val, location_array_val, location_array_ev_val = load_gt.load(VALID_GT, N_VAL, 3)
    if PREPRO_MODE:
        # Load and pre-process received signals
        h_rad_train, h_rad_val  = LoadPreProcess(HP["K_C"], HP["PP"], 0.0, 0.0, HP["SR_C"])
        # detect MultiPath Components
        previousGEO.DetectMPC(DATA_PATH, HP["K_C"], SNR_LIST, PREPRO_SAVE_MODE, h_rad_train, count_array_train, "train")
        previousGEO.DetectMPC(DATA_PATH, HP["K_C"], SNR_LIST, PREPRO_SAVE_MODE, h_rad_val, count_array_val, "val")
    # Training and testing for counting/localization
    pred_c_train, pred_c    = previousGEO.Counting(DATA_PATH, SNR_LIST, count_array_train, N_TRAIN, N_VAL)
    pred_l_train, pred_l    = previousGEO.Localization(DATA_PATH, SNR_LIST, N_TRAIN, N_VAL, pred_c_train, pred_c)
    # Report accuracy
    c_train_GEO, l_train_GEO, c_val_GEO, l_val_GEO = Report(count_array_train, location_array_ev_train, count_array_val, location_array_ev_val, pred_c_train, pred_l_train, pred_c, pred_l, HP["MT"])

### Graph
fig = plt.figure(figsize=[30,5])
ax  = []
for ax_i in range(1,4):
    ax.append(fig.add_subplot(1,3,ax_i))

# Counting Accuracy
if SOLUTION_MODE["ViT"]:
    ax[0].plot(SNR_LIST, c_val_ViT, 'bo-', label='proposed (ViT)')
if SOLUTION_MODE["CNN"]:
    ax[0].plot(SNR_LIST, c_val_CNN, 'ro-',  label='proposed (CNN)')
if SOLUTION_MODE["DNN"]:
    ax[0].plot(SNR_LIST, c_val_DNN, 'go--',  label='DNN-based')
if SOLUTION_MODE["GEO"]:
    ax[0].plot(SNR_LIST, c_val_GEO, 'yo--',  label='geometry-based')
ax[0].legend()
ax[0].set_xlabel('SNR [dB]')
ax[0].set_ylabel('Counting Accuracy')
ax[0].set_title('Counting Accuracy')
ax[0].set_xlim([-20, 20])
ax[0].set_ylim([0.0, 1.0])
ax[0].grid(True)

# Localization Accuracy
if SOLUTION_MODE["ViT"]:
    ax[1].plot(SNR_LIST, [l_val_ViT[0], l_val_ViT[9], l_val_ViT[18]], 'bo-', label='proposed (ViT)')
if SOLUTION_MODE["CNN"]:
    ax[1].plot(SNR_LIST, [l_val_CNN[0], l_val_CNN[9], l_val_CNN[18]], 'ro-',  label='proposed (CNN)')
if SOLUTION_MODE["DNN"]:
    ax[1].plot(SNR_LIST, [l_val_DNN[0], l_val_DNN[9], l_val_DNN[18]], 'go--',  label='DNN-based')
if SOLUTION_MODE["GEO"]:
    ax[1].plot(SNR_LIST, [l_val_GEO[0], l_val_GEO[9], l_val_GEO[18]], 'yo--',  label='geometry-based')
ax[1].legend()
ax[1].set_xlabel('SNR [dB]')
ax[1].set_ylabel('Localization Accuracy')
ax[1].set_title('Localization Accuracy')
ax[1].set_xlim([-20, 20])
ax[1].set_ylim([0.0, 0.5])
ax[1].grid(True)

# Localization Accuracy for each number of persons at SNR=18dB
if SOLUTION_MODE["ViT"]:
    ax[2].plot(list(range(1,9)), l_val_ViT[19:27], 'bo-', label='proposed (ViT)')
if SOLUTION_MODE["CNN"]:
    ax[2].plot(list(range(1,9)), l_val_CNN[19:27], 'ro-',  label='proposed (CNN)')
if SOLUTION_MODE["DNN"]:
    ax[2].plot(list(range(1,9)), l_val_DNN[19:27], 'go--',  label='DNN-based')
if SOLUTION_MODE["GEO"]:
    ax[2].plot(list(range(1,9)), l_val_GEO[19:27], 'yo--',  label='geometry-based')
ax[2].legend()
ax[2].set_xlabel('number of persons')
ax[2].set_ylabel('Localization Accuracy')
ax[2].set_title('Details of Localization Accuracy (SNR=18dB)')
ax[2].set_xlim([1,8])
ax[2].set_ylim([0.0, 1.0])
ax[2].grid(True)

plt.show()
