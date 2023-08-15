
### Module to evaluate, print, save the accuracy result

import numpy as np

# Fixed parameters
MAX_PERSONS = 8
NUM_SECTORS = 9

# evaluate
def evaluate(count_array_train, location_array_ev_train, count_array_val, location_array_ev_val, pred_c_train, pred_l_train, pred_c, pred_l):
    
    NUM_SNR = len(pred_c_train)
    n_train = len(count_array_train)
    n_val   = len(count_array_val)
    c_train_w, l_train_w, c_val_w, l_val_w = [], [], [], []
    
    for snr_idx in range(NUM_SNR):
    
        count_train_w           = 0
        count_train_w_sum       = 0
        localization_train_w    = 0
        localization_train_w_sum= 0
        localization_train_sum_array = np.zeros((MAX_PERSONS))
        localization_train_array     = np.zeros((MAX_PERSONS))
        
        count_val_w             = 0
        count_val_w_sum         = 0
        localization_val_w      = 0
        localization_val_w_sum  = 0
        localization_val_sum_array = np.zeros((MAX_PERSONS))
        localization_val_array     = np.zeros((MAX_PERSONS))
        
        for data_i in range(n_train):
            c_v = int(np.argmax(count_array_train[data_i])+1)
            count_train_w_sum += c_v
            localization_train_w_sum += c_v
            localization_train_sum_array[int(c_v-1)] += 1
            if pred_c_train[snr_idx][data_i] == c_v:
                count_train_w += c_v
            if (pred_l_train[snr_idx][data_i] == location_array_ev_train[data_i]).all():
                localization_train_w += c_v
                localization_train_array[int(c_v-1)] += 1
                
        for data_i in range(n_val):
            c_v = int(np.argmax(count_array_val[data_i])+1)
            count_val_w_sum += c_v
            localization_val_w_sum += c_v
            localization_val_sum_array[int(c_v-1)] += 1
            if pred_c[snr_idx][data_i] == c_v:
                count_val_w += c_v
            if (pred_l[snr_idx][data_i] == location_array_ev_val[data_i]).all():
                localization_val_w += c_v
                localization_val_array[int(c_v-1)] += 1
                
        c_train_w.append(count_train_w/count_train_w_sum)
        l_train_w.append(localization_train_w/localization_train_w_sum)
        l_train_w = l_train_w + list(localization_train_array/localization_train_sum_array)
        c_val_w.append(count_val_w/count_val_w_sum)
        l_val_w.append(localization_val_w/localization_val_w_sum)
        l_val_w = l_val_w + list(localization_val_array/localization_val_sum_array)
    
    return c_train_w, l_train_w, c_val_w, l_val_w


#function sector_form : Funtion to convert model predictions into sector format string
# Where:
#    Input :
#       pred_l: prediction class labels from ML model for localization
#       SNR_LIST: snr lsit value
#   Output: It creates a .txt file to store sector format strings
def save_sector_form(pred_l, SNR_LIST, filename):
    
    for snr_idx, snr_val in enumerate(SNR_LIST):
        n_val   = len(pred_l[snr_idx])
        lin_all = ""
        for data_i in range(n_val):
            
            lin = ""
            for sector_i in range(NUM_SECTORS):
                num_person_sector = int(pred_l[snr_idx][data_i, sector_i])
                for person_i in range(num_person_sector):
                    lin += chr(ord("A") + sector_i)
            lin_all += lin + "\n"

        f = open("output/" + filename + str(snr_val) + '.txt', 'w')
        f.write(lin_all)
        f.close()

# Report
def print_result(c_train_w, l_train_w, c_val_w, l_val_w, SNR_LIST):
    print("SNR[dB]" + "\t" + "count(training)" + "\t" + "locate(training)" + "\t" + "count(test)" + "\t" + "locate(test)")
    for snr_idx, snr_val in reversed(list(enumerate(SNR_LIST))):
        SNR         = "{:3.0f}".format(snr_val) #str(snr_val)
        count_t     = "\t" +        "{:9.2f}".format(c_train_w[snr_idx] * 100)
        locate_t    = "\t\t" +      "{:9.2f}".format(l_train_w[snr_idx * (MAX_PERSONS + 1)] * 100)
        count_v     = "\t\t\t" +    "{:9.2f}".format(c_val_w[snr_idx] * 100)
        locate_v    = "\t" +        "{:9.2f}".format(l_val_w[snr_idx * (MAX_PERSONS + 1)] * 100)
        print(SNR + count_t + locate_t + count_v + locate_v)
        
