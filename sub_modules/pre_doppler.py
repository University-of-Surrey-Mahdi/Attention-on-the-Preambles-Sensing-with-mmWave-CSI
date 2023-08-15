### Doppler domain analysis at pre-process

import numpy as np

# Fixed parameters
N_L             = 44
N_RX            = 4
N_TX            = 4
N_CH            = 128

def process(H, mean_k):
    
    H_VEL = np.zeros((N_RX, N_TX, (N_L + 1), int(N_CH/mean_k)), dtype=np.complex128) # [N_RX, N_TX, (N_L + 1), N_CH/K]
    h_slice = np.zeros((int(N_CH/mean_k), 1), dtype=np.complex128)
        
    for tap in range(N_L + 1):
        for tx_i in range(N_TX):
            for rx_i in range(N_RX):
                for ch_i in range(int(N_CH/mean_k)):
                    h_slice[ch_i] = np.mean(H[rx_i, tx_i, tap, mean_k * ch_i : mean_k * (ch_i + 1)])
                fftoutput = np.fft.fft(np.squeeze(h_slice))[::-1]
                H_VEL[rx_i, tx_i, tap, :] = np.roll(fftoutput, 1)
      
    return H_VEL # [N_RX, N_TX, (N_L + 1), N_CH/K]
