# Conventional geometry-based method 

import os
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import pickle 
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import torch
# import required modules to output prediction result
import sub_modules.post as post
# define device type
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

THRESHOLD   = 2
CLUSTERING  = 2
N_GOLAY     = 512
N_CH        = 128
MAX_PERSONS = 8
MAX_SECTOR  = 9 
N_RX        = 4         # Number of antennas at RX device (4)
N_TX        = 4         # Number of antennas at TX device (4)
LAM         = 1         # Wavelength
D           = LAM / 2   # antenna spacing
PHI_RES     = 1000      # Phi resolution of antenna pattern
COR_RES     = 20        # coordinate (x, y, z) resolution
LIGHT_SPEED = 299792458 # Light speed [m/s]
FREQ        = 1.76e9    # Sampling rate [s/s]
DIS_UNIT    = LIGHT_SPEED / FREQ # Distance unit
DIS_MIN     = 10        # Distance of direct path
TX          = np.array([0,3.5,2.8])     # [x, y, z] of TX device
RX          = np.array([7,3.5,2.8])     # [x, y, z] of RX device


def gain(d, w):
    """Return the power as a function of azimuthal angle, phi."""
    phi = np.linspace(0, 2*np.pi, PHI_RES)
    psi = 2*np.pi * d / LAM * np.cos(phi)
    j = np.arange(len(w))
    A = np.sum(w[j] * np.exp(j * 1j * psi[:, None]), axis=1)
    g = np.abs(A)**2
    return phi, g


def sector_detect(x,y):
    if x < 7/3:
        # A,B,C
        if y > 7/3*2:
            ans = 1
        elif y > 7/3*1:
            ans = 2
        else:
            ans = 3
    elif x < 7/3*2:
        # D,E,F
        if y > 7/3*2:
            ans = 4
        elif y > 7/3*1:
            ans = 5
        else:
            ans = 6
    else:
        # G,H,I
        if y > 7/3*2:
            ans = 7
        elif y > 7/3*1:
            ans = 8
        else:
            ans = 9
    
    return ans

def near_sector_D(x,y):
    if y < x:
        if -y+3.5 < x:
            return np.array([2,3])
        else:
            return np.array([3,3])
    else:
        if -y+3.5 < x:
            return np.array([2,2])
        else:
            return np.array([3,2])
        
# Geometry-based dictionary 
for i in range(N_RX):
    w = np.zeros(N_TX)*1j
    for j in range(N_TX):
        w[j] = np.exp(-2*np.pi*1j / N_TX * (i+1) * (j+1))
    # Calculate gain and directive gain; plot on a polar chart.
    phi, g = gain(D, w)

# Dictionary of [x, y, z] ==>  [rx_idx, tx_idx, distance, sector]
phi_index_array = np.linspace(0, 2*np.pi, PHI_RES)
ans_array = np.zeros((COR_RES, COR_RES, COR_RES, 4)) # x, y, z, (rx, tx, distance, sector)
for x_i in range(COR_RES):
    x = x_i / COR_RES * 7
    for y_i in range(int(COR_RES/2)):
        y = y_i / COR_RES * 7 + 3.5
        for z_i in range(COR_RES):
            z = z_i / COR_RES * 2
            
            # determine [sector]
            sector_detected = sector_detect(x,y)
            ans_array[x_i,y_i,z_i,3] = sector_detected
            
            # determine [rx idx, tx idx] from histogram analysis of training data
            if sector_detected == 1 or sector_detected == 3:    # A/C
                ans_array[x_i,y_i,z_i,0] = 3
                ans_array[x_i,y_i,z_i,1] = 2
            elif sector_detected == 2:                          # B
                ans_array[x_i,y_i,z_i,0] = 3
                ans_array[x_i,y_i,z_i,1] = 1
            elif sector_detected == 4 or sector_detected == 6:  # D/F
                if sector_detected == 4:                        # D
                    ans_array[x_i,y_i,z_i,:2] = near_sector_D(x,y)
                elif sector_detected == 6:                      # F
                    ans_array[x_i,y_i,z_i,:2] = near_sector_D(x,7-y)
                else:
                    print("PROBLEM")
            elif sector_detected == 5:                          # E
                ans_array[x_i,y_i,z_i,0] = 3
                ans_array[x_i,y_i,z_i,1] = 3
            elif sector_detected == 7 or sector_detected == 9:  # G/I
                ans_array[x_i,y_i,z_i,0] = 2
                ans_array[x_i,y_i,z_i,1] = 3
            elif sector_detected == 8:                          # H
                ans_array[x_i,y_i,z_i,0] = 1
                ans_array[x_i,y_i,z_i,1] = 3
            else:
                print("PROBLEM")
                    
            # determine [distance]
            distance = np.linalg.norm(TX - np.array([x,y,z])) + np.linalg.norm(RX - np.array([x,y,z]))
            ans_array[x_i,y_i,z_i,2] = int(round((distance - 7) / DIS_UNIT + DIS_MIN))
            
            

def save_dict(parent_path, snr_val, clustering, file_name, dict_save):
    folder = parent_path + str(snr_val) + "_feature_power_cluster_" + str(clustering)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    with open(folder + "/" + file_name + ".pkl", 'wb') as f:
        pickle.dump(dict_save, f)

def load_dict(parent_path, snr_val, clustering, file_name):
    folder = parent_path + str(snr_val) + "_feature_power_cluster_" + str(clustering)
    with open(folder + "/" + file_name + ".pkl", 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def detect_peaks(image, i, j, clustering):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    num_peaks = detected_peaks.sum()
    
    # peak list
    MAX_CLUSTER = int(num_peaks / clustering)
    peak_list_woant = np.argwhere(detected_peaks == True)
    peak_list = np.zeros((num_peaks,5))
    peak_list[:, 0] = i
    peak_list[:, 1] = j
    peak_list[:, 2:4] = peak_list_woant
    peak_list[:, 4] = (image * image)[detected_peaks]
    
    kmeans  = KMeans(n_clusters=MAX_CLUSTER, random_state=42, n_init="auto").fit(peak_list_woant, sample_weight=peak_list[:, 4])
    cluster = kmeans.predict(peak_list_woant)
    
    num_clusters = cluster.max() + 1
    peak_list_cluster = np.zeros((num_clusters,5))
    peak_list_cluster[:, 0] = i
    peak_list_cluster[:, 1] = j
    peak_list_cluster[:, 2:4] = kmeans.cluster_centers_
    for num_clusters_i in range(num_clusters):
        sample_index = np.argwhere(cluster == num_clusters_i)
        peak_list_cluster[num_clusters_i, 4] = peak_list[sample_index,4].sum()
    
    return num_peaks, peak_list_cluster     # int, [rx_idx, tx_idx, distance, velocity, received_power]
    
def peak_hist(thre, clustering, h_rad_local):
    over_thre = (h_rad_local > thre)
    h_rad_local_energy_over = (h_rad_local[over_thre] * h_rad_local[over_thre])
    h_rad_local_replace     = h_rad_local
    h_rad_local_replace[(h_rad_local_replace <= thre)] = 0
   
    if over_thre.sum() == 0:
        h_energy_total  = 0
    else:
        h_energy_total  = h_rad_local_energy_over.sum()
    
    h_num_peaks     = 0
    FIRST = True
    for i in range(4):
        for j in range(4):
            num_peaks, peak_list_tmp    = detect_peaks(h_rad_local_replace[i,j,:,:], i, j, clustering)  # int, [rx_idx, tx_idx, distance, velocity, received_power]
            h_num_peaks += num_peaks
            # print(i, j, detected_peaks.sum())
            if FIRST:
                peak_list = peak_list_tmp
                FIRST = False
            else:
                peak_list = np.concatenate([peak_list, peak_list_tmp])
    
    return h_energy_total, h_num_peaks, peak_list   # float, int, [[rx_idx, tx_idx, distance, velocity, received_power]]


def DetectMPC(parent_path, mean_k, SNR_LIST, PREPRO_SAVE_MODE, h_rad, count_array, trval):
    num_sample = int(len(h_rad) / len(SNR_LIST))
    
    for snr_idx, snr_val in enumerate(SNR_LIST):
        thre_noise = THRESHOLD * np.sqrt((1/pow(10, snr_val/10))/N_GOLAY * N_CH/pow(mean_k,2))
        
        dict_e_total  = {}
        dict_n_peaks  = {}
        dict_l_peak   = {}
        
        for c_v_i in range(1,MAX_PERSONS+1):
            dict_e_total[c_v_i]    = []
            dict_n_peaks[c_v_i]    = []
            dict_l_peak[c_v_i]     = []
            
        for hf_i in range(num_sample):
            
            # float, int, [[rx_idx, tx_idx, distance, velocity, received_power]]
            h_energy_total, h_num_peaks, peak_list = peak_hist(thre_noise, CLUSTERING, h_rad[(hf_i + snr_idx * num_sample),:,:,:,1:])
            c_v = int(np.argmax(count_array[hf_i])+1)
            
            dict_e_total[c_v].append([h_energy_total,hf_i])
            dict_n_peaks[c_v].append([h_num_peaks,hf_i])
            dict_l_peak[c_v].append([peak_list,hf_i])
                                
        if PREPRO_SAVE_MODE:
            save_dict(parent_path, snr_val, CLUSTERING,  "dict_e_total_" + trval, dict_e_total)
            save_dict(parent_path, snr_val, CLUSTERING,  "dict_n_peaks_" + trval, dict_n_peaks)
            save_dict(parent_path, snr_val, CLUSTERING,  "dict_l_peak_" + trval, dict_l_peak)



def Counting(parent_path, SNR_LIST, count_array_train, N_TRAIN, N_VAL):
    
    # prediction result
    pred_c_train    = []
    pred_c          = []
    
    for SNR_VAL in SNR_LIST:
    
        # load X values
        dict_e_total_train      = load_dict(parent_path, SNR_VAL, CLUSTERING, "dict_e_total_train")
        dict_n_peaks_train      = load_dict(parent_path, SNR_VAL, CLUSTERING, "dict_n_peaks_train")
        dict_e_total_val        = load_dict(parent_path, SNR_VAL, CLUSTERING, "dict_e_total_val")
        dict_n_peaks_val        = load_dict(parent_path, SNR_VAL, CLUSTERING, "dict_n_peaks_val")
        
        ## data prepare for counting 
        X_train = np.zeros((N_TRAIN,2))
        X_test  = np.zeros((N_VAL,2))
        y_train = np.zeros((N_TRAIN))
        y_test  = np.zeros((N_VAL))
        
        for key, value in dict_n_peaks_train.items():
            for i in value:
                if N_TRAIN > i[1]:
                    X_train[i[1],0] = i[0]      # X_train[hf_i, 0] = h_num_peaks
                    y_train[i[1]] = key - 1     # Counting (0 - 7)
        
        for key, value in dict_e_total_train.items():
            for i in value:
                if N_TRAIN > i[1]:
                    X_train[i[1],1] = i[0]      # X_train[hf_i, 1] = h_energy_total
                
        for key, value in dict_n_peaks_val.items():
            for i in value:
                if N_VAL > i[1]:
                    X_test[i[1],0] = i[0]       # X_test[hf_i, 0] = h_num_peaks
                    y_test[i[1]] = key - 1      # Counting (0 - 7)
        
        for key, value in dict_e_total_val.items():
            for i in value:
                if N_VAL > i[1]:
                    X_test[i[1],1] = i[0]       # X_test[hf_i, 1] = h_energy_total
        
        # Counting by SVM
        model = SVC(C=0.1)
        model.fit(X_train, y_train) # SVM training
        predicted_train = model.predict(X_train)    # prediction by SVM
        predicted       = model.predict(X_test)     # prediction by SVM
        
        # Store the prediction result
        pred_c_train.append( predicted_train + 1 )
        pred_c.append( predicted + 1 )
        
    return pred_c_train, pred_c
    

def detect_position(L_trval, reg, pred_c):
    np.random.seed(42)
    Y_pred_tr = np.zeros((int(len(L_trval)), MAX_SECTOR), dtype=np.float32)
    
    # Set post-process configures
    post.set_param(MAX_PERSONS, MAX_SECTOR)
    
    for ii, peak_list in enumerate(L_trval):
        # peak_list = [[rx_idx, tx_idx, distance, velocity, received_power]]
        
        peak_list[:,2] = np.round(peak_list[:,2])   # rounding the distance
        sector = np.array([]).reshape(0,2)          # detected sectors [[sector, power]]
        
        for peak_list_i in range(len(peak_list)):
            
            # [rx_idx, tx_idx, distance] ==> int key
            current_key = peak_list[peak_list_i][0]*1000+peak_list[peak_list_i][1]*100+peak_list[peak_list_i][2]
            
            if current_key in reg.keys():
                # The key is already registered
                if reg[current_key] != False:
                    # correct key
                    sector = np.concatenate([sector, np.array([[reg[current_key][3],peak_list[peak_list_i][4]]])])
                else:
                    # incorrect key
                    pass
            else:
                # Need to find and register [x, y, z, sector] at currenct key 
                cand = np.array([]).reshape(0,3)    # candidates of coordinate [[x, y, z]]
                for x_i in range(COR_RES):
                    x = x_i / COR_RES * 7
                    for y_i in range(int(COR_RES/2)):
                        y = y_i / COR_RES * 7 + 3.5
                        for z_i in range(COR_RES):
                            z = z_i / COR_RES * 2
                            
                            # Seek candidates of coordinate from registered "ans_array"
                            if (ans_array[x_i,y_i,z_i,:3] == peak_list[peak_list_i][:3]).all():
                                cand = np.concatenate([cand, np.array([[x,y,z]])])
                
                if len(cand) == 0:
                    # incorrect key
                    reg[current_key] = False
                else:
                    # candidates of coordinate have been found
                    # determine the sector from the center of the gravity
                    cor_x, cor_y, cor_z = cand[:,0].mean(), cand[:,1].mean(), cand[:,2].mean()
                    sector = np.concatenate([sector, np.array([[sector_detect(cor_x,cor_y),peak_list[peak_list_i][4]]])])
                    
                    # [rx_idx, tx_idx, distance] ==> [x, y, z, sector] 
                    reg[current_key] = [cor_x, cor_y, cor_z] + [sector_detect(cor_x,cor_y)]
        
        # prediction from probability density
        pred_localization = np.zeros((MAX_SECTOR)) # probability density (power density)
        for i in range(1,(MAX_SECTOR+1)):
            pred_localization[i-1] = sector[sector[:,0] == i][:,1].sum() / sector[:,1].sum()

        pred_tr_l       = post.pred((torch.from_numpy(np.array([pred_localization]))).to(device), (torch.from_numpy(np.array([pred_c[ii]]))).to(device))[0]
        
        # swap (A<=>C, D<=>F, G<=>I) randomly
        pred_tr_l       = pred_tr_l.detach().cpu().numpy()
        pred_tr_l_swap  = pred_tr_l.copy()
        for i in [0,3,6]:
            for j in range(int(pred_tr_l[i])):
                if np.random.rand() > 0.5:
                    pred_tr_l_swap[i] -= 1
                    pred_tr_l_swap[int(i+2)] += 1
        
        Y_pred_tr[ii] = pred_tr_l_swap
    
    return Y_pred_tr
    
    
def Localization(parent_path, SNR_LIST, N_TRAIN, N_VAL, pred_c_train, pred_c):
    
    # prediction result
    pred_l_train    = []
    pred_l          = []
    
    # Dictionary of [rx_idx, tx_idx, distance] ==> [x, y, z, sector] 
    reg = {}
    
    for SNR_IDX, SNR_VAL in enumerate(SNR_LIST):
    
        # load X values
        dict_l_peak_train       = load_dict(parent_path, SNR_VAL, CLUSTERING, "dict_l_peak_train")
        dict_l_peak_val         = load_dict(parent_path, SNR_VAL, CLUSTERING, "dict_l_peak_val")
        
        ## data prepare for localization
        L_train = list(range(N_TRAIN))
        L_test  = list(range(N_VAL))
        for key, value in dict_l_peak_train.items():
            for i in value:
                if N_TRAIN > i[1]:
                    L_train[i[1]] = i[0][:,:]   # L_train[hf_i] = [[rx_idx, tx_idx, distance, velocity, received_power]]
        for key, value in dict_l_peak_val.items():
            for i in value:
                if N_VAL > i[1]:
                    L_test[i[1]] = i[0][:,:]    # L_test[hf_i] = [[rx_idx, tx_idx, distance, velocity, received_power]]
        
        # prediction result
        Y_pred_tr_train = detect_position(L_train, reg, pred_c_train[SNR_IDX])
        Y_pred_tr       = detect_position(L_test,  reg, pred_c[SNR_IDX])
        
        pred_l_train.append( Y_pred_tr_train )
        pred_l.append( Y_pred_tr )
        
    return pred_l_train, pred_l




