### CH estimation at pre-process

import numpy as np
from scipy.io import loadmat

# Transmitted signal
tx = loadmat("sub_modules/tx.mat")['tx']
Tx = np.transpose(tx)

# Fixed parameters
N_SYM           = 1024
N_L             = 44
N_RX            = 4
N_TX            = 4
N_CH            = 128
START_FIELD2    = 1280


# X matrix representing path latency
def X_signal(start_field):
    x_slice = np.zeros((N_TX, N_SYM + N_L))
    x_slice[:, :N_SYM] = Tx[:, start_field:start_field+N_SYM]
    X = x_slice
    for j in range(N_L):     
        x_slice = np.roll(x_slice, 1, axis=1)
        X = np.vstack((X, x_slice))
    return X

# X signals
X1 = X_signal(0)            # Field-1, [N_TX * (L + 1), (N_SYM + N_L)]
X2 = X_signal(START_FIELD2) # Field-2, [N_TX * (L + 1), (N_SYM + N_L)]

# (X * X^T)^-1
x_xt_inv    = np.linalg.inv(np.matmul(X1, np.transpose(X1)) + np.matmul(X2, np.transpose(X2))) # [N_TX * (L + 1), N_TX * (L + 1)]

# split array
v_arr   = np.array(range(N_TX, N_TX * (N_L + 1), N_TX), dtype=np.int64)


def LSE(rx):
    
    H = np.zeros((N_RX, N_TX, (N_L + 1), N_CH), dtype=np.complex128) # [N_RX, N_TX, (N_L + 1), N_CH]
    
    # extract IEEE 802.11ay EDMG-CEF Golay sequence
    Y1 = rx[:(N_SYM + N_L), :,:]                            # [(N_SYM + N_L), N_RX, N_CH]
    Y2 = rx[START_FIELD2:(START_FIELD2 + N_SYM + N_L), :,:] # [(N_SYM + N_L), N_RX, N_CH]
    Y1 = np.einsum('ijk->jik', Y1)                          # [N_RX, (N_SYM + N_L), N_CH]
    Y2 = np.einsum('ijk->jik', Y2)                          # [N_RX, (N_SYM + N_L), N_CH]
    
    for ch_i in range(N_CH):
        Y1_eachCH  = Y1[:, :, ch_i]     # [N_RX, (N_SYM + N_L)]
        Y2_eachCH  = Y2[:, :, ch_i]     # [N_RX, (N_SYM + N_L)]
        
        # Y * X^T
        y_xt = np.matmul(Y1_eachCH, np.transpose(X1)) + np.matmul(Y2_eachCH, np.transpose(X2)) # [N_RX, N_TX * (N_L + 1)]
        
        # CH estimated by LSE
        H_eachCH = np.matmul(y_xt, x_xt_inv)            # [N_RX, N_TX * (N_L + 1)]
        H_eachCH = np.array(np.hsplit(H_eachCH, v_arr)) # [(N_L + 1), N_RX, N_TX]
        H_eachCH = np.einsum('ijk->jki', H_eachCH)      # [N_RX, N_TX, (N_L + 1)]
        
        # Store each H to H
        H[:, :, :, ch_i] = H_eachCH

    return H # [N_RX, N_TX, (N_L + 1), N_CH]


def LSE_woSplit(rx):
    
    H_woSplit = np.zeros((N_RX, ((N_L + 1) * N_TX), N_CH), dtype=np.complex128) # [N_RX, ((N_L + 1) * N_TX), N_CH]
    
    # extract IEEE 802.11ay EDMG-CEF Golay sequence
    Y1 = rx[:(N_SYM + N_L), :,:]                            # [(N_SYM + N_L), N_RX, N_CH]
    Y2 = rx[START_FIELD2:(START_FIELD2 + N_SYM + N_L), :,:] # [(N_SYM + N_L), N_RX, N_CH]
    Y1 = np.einsum('ijk->jik', Y1)                          # [N_RX, (N_SYM + N_L), N_CH]
    Y2 = np.einsum('ijk->jik', Y2)                          # [N_RX, (N_SYM + N_L), N_CH]
    
    for ch_i in range(N_CH):
        Y1_eachCH  = Y1[:, :, ch_i]     # [N_RX, (N_SYM + N_L)]
        Y2_eachCH  = Y2[:, :, ch_i]     # [N_RX, (N_SYM + N_L)]
        
        # Y * X^T
        y_xt = np.matmul(Y1_eachCH, np.transpose(X1)) + np.matmul(Y2_eachCH, np.transpose(X2)) # [N_RX, N_TX * (N_L + 1)]
        
        # CH estimated by LSE
        H_eachCH = np.matmul(y_xt, x_xt_inv)            # [N_RX, N_TX * (N_L + 1)]
        
        # Store each H to H
        H_woSplit[:, :, ch_i] = H_eachCH

    return H_woSplit # [N_RX, ((N_L + 1) * N_TX), N_CH]


def CalcCOV(path_rs, path_cov, snr_val, NumTestSamples):
    # Calculate Covariance for LMMSE from 18dB data
    
    Y1_conv     = np.zeros((N_RX, (N_SYM + N_L), (N_SYM + N_L)), dtype=np.complex128)
    Y2_conv     = np.zeros((N_RX, (N_SYM + N_L), (N_SYM + N_L)), dtype=np.complex128)
    Y1_h_conv   = np.zeros((N_RX, (N_SYM + N_L), (N_SYM + N_L)), dtype=np.complex128)
    Y2_h_conv   = np.zeros((N_RX, (N_SYM + N_L), (N_SYM + N_L)), dtype=np.complex128)
    Y1_mean     = np.zeros((N_RX, (N_SYM + N_L)), dtype=np.complex128)
    Y2_mean     = np.zeros((N_RX, (N_SYM + N_L)), dtype=np.complex128)
    h_f_mean    = np.zeros((N_RX, ((N_L + 1) * N_TX)), dtype=np.complex128)
    
    for N_r_i in range(N_RX):
    
    
        Y1  = np.zeros(((N_SYM + N_L), N_CH * NumTestSamples), dtype=np.complex128)
        Y2  = np.zeros(((N_SYM + N_L), N_CH * NumTestSamples), dtype=np.complex128)
        h_f = np.zeros(((N_SYM + N_L), N_CH * NumTestSamples), dtype=np.complex128)
        
        for i in range(NumTestSamples):
            
            # read received signal (.mat)
            rx = loadmat(path_rs + "snr" + str(snr_val) + '/rxSigCh' + str(i) + '.mat')['rxNoisy'] # [received signal, N_RX, N_CH] 
            
            h_f_local, Y1_local, Y2_local = LSE_woSplit(rx) 
            
            Y1[:,N_CH*i:N_CH*(i+1)] = Y1_local[N_r_i]
            Y2[:,N_CH*i:N_CH*(i+1)] = Y2_local[N_r_i]
            h_f[:,N_CH*i:N_CH*(i+1)] = h_f_local[N_r_i]
            
        Y1_conv[N_r_i]      = np.cov(Y1)
        Y2_conv[N_r_i]      = np.cov(Y2)
        Y1_h_conv[N_r_i]    = np.cov(Y1, h_f)[(N_SYM + N_L):,:(N_SYM + N_L)]
        Y2_h_conv[N_r_i]    = np.cov(Y2, h_f)[(N_SYM + N_L):,:(N_SYM + N_L)]
        
        Y1_mean[N_r_i]      = Y1.mean(axis=1)
        Y2_mean[N_r_i]      = Y2.mean(axis=1)
        h_f_mean[N_r_i]     = h_f.mean(axis=1)
        
    with open(path_cov + 'Y1_conv.npy', 'wb') as f:
        np.save(f, Y1_conv)
    with open(path_cov + 'Y2_conv.npy', 'wb') as f:
        np.save(f, Y2_conv)
    with open(path_cov + 'Y1_h_conv.npy', 'wb') as f:
        np.save(f, Y1_h_conv)
    with open(path_cov + 'Y2_h_conv.npy', 'wb') as f:
        np.save(f, Y2_h_conv)
    with open(path_cov + 'Y1_mean.npy', 'wb') as f:
        np.save(f, Y1_mean)
    with open(path_cov + 'Y2_mean.npy', 'wb') as f:
        np.save(f, Y2_mean)
    with open(path_cov + 'h_f_mean.npy', 'wb') as f:
        np.save(f, h_f_mean)
    
    return Y1_conv, Y2_conv, Y1_h_conv, Y2_h_conv, Y1_mean, Y2_mean, h_f_mean


def LMMSE(path_cov, rx):
    
    # Load covariance matrices    
    with open(path_cov + 'Y1_conv.npy', 'rb') as f:
        Y1_conv = np.load(f)
    with open(path_cov + 'Y2_conv.npy', 'rb') as f:
        Y2_conv = np.load(f)
    with open(path_cov + 'Y1_h_conv.npy', 'rb') as f:
        Y1_h_conv = np.load(f)
    with open(path_cov + 'Y2_h_conv.npy', 'rb') as f:
        Y2_h_conv = np.load(f)
    with open(path_cov + 'Y1_mean.npy', 'rb') as f:
        Y1_mean = np.load(f)
    with open(path_cov + 'Y2_mean.npy', 'rb') as f:
        Y2_mean = np.load(f)
    with open(path_cov + 'h_f_mean.npy', 'rb') as f:
        h_f_mean = np.load(f)
    
    Y1_conv_mat  = np.zeros(Y1_h_conv.shape, dtype=np.complex128)
    Y2_conv_mat  = np.zeros(Y2_h_conv.shape, dtype=np.complex128)
    
    for N_r_i in range(N_RX):
        Y1_conv_mat[N_r_i]  = np.matmul(Y1_h_conv[N_r_i], np.linalg.inv(Y1_conv[N_r_i]))
        Y2_conv_mat[N_r_i]  = np.matmul(Y2_h_conv[N_r_i], np.linalg.inv(Y2_conv[N_r_i]))    
    
    
    H = np.zeros((N_RX, N_TX, (N_L + 1), N_CH), dtype=np.complex128) # [N_RX, N_TX, (N_L + 1), N_CH]
    
    # extract IEEE 802.11ay EDMG-CEF Golay sequence
    Y1 = rx[:(N_SYM + N_L), :,:]                            # [(N_SYM + N_L), N_RX, N_CH]
    Y2 = rx[START_FIELD2:(START_FIELD2 + N_SYM + N_L), :,:] # [(N_SYM + N_L), N_RX, N_CH]
    Y1 = np.einsum('ijk->jik', Y1)                          # [N_RX, (N_SYM + N_L), N_CH]
    Y2 = np.einsum('ijk->jik', Y2)                          # [N_RX, (N_SYM + N_L), N_CH]
    
    for ch_i in range(N_CH):
        Y1_eachCH  = Y1[:, :, ch_i]     # [N_RX, (N_SYM + N_L)]
        Y2_eachCH  = Y2[:, :, ch_i]     # [N_RX, (N_SYM + N_L)]
        
        for N_r_i in range(N_RX):
            
            h_f_1 = h_f_mean[N_r_i] + np.matmul(Y1_conv_mat[N_r_i], (Y1_eachCH[N_r_i] - Y1_mean[N_r_i]))
            h_f_2 = h_f_mean[N_r_i] + np.matmul(Y2_conv_mat[N_r_i], (Y2_eachCH[N_r_i] - Y2_mean[N_r_i]))
            h_f_m = (h_f_1 + h_f_2)/2
            h_f_m_split = np.einsum('ij->ji', np.array(np.hsplit(h_f_m, v_arr)))
            H[N_r_i,:,:,ch_i] = h_f_m_split

    return H # [N_RX, N_TX, (N_L + 1), N_CH]
         