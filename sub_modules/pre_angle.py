### Angular domain analysis at pre-process

import numpy as np

# Fixed parameters
N_L             = 44
N_RX            = 4
N_TX            = 4
N_CH            = 128

# Variable parameters, but set the default
def unitary_matrix(n_matrix, theta):
    unitary = np.zeros((n_matrix, n_matrix), dtype=np.complex64)
    for k in range(1, (n_matrix + 1)):
        for l in range(1, (n_matrix + 1)):     
            unitary[k-1, l-1] = (1 / np.sqrt(n_matrix)) * (np.exp(-1j * 2 * np.pi * (k + theta) * l / n_matrix))
    return unitary

THETA_R = 0.2
THETA_T = 0.8
UR_TR   = unitary_matrix(N_RX, THETA_R)
UT      = unitary_matrix(N_TX, THETA_T)
UT      = np.matrix.getH(UT)


# Set variable parameters
def set_phase_shift(theta_r, theta_t):
    global THETA_R
    global THETA_T
    global UR_TR
    global UT
    
    THETA_R = theta_r
    THETA_T = theta_t
    UR_TR   = unitary_matrix(N_RX, THETA_R)
    UT      = unitary_matrix(N_TX, THETA_T)
    UT      = np.matrix.getH(UT)

# Angular domain analysis
def process(H_VEL, mean_k):
    
    H_VEL_ANG   = np.zeros((N_RX, N_TX, (N_L + 1), int(N_CH/mean_k)), dtype=np.complex128) # [N_RX, N_TX, (N_L + 1), N_CH/K]
        
    for ch_i in range(int(N_CH/mean_k)):
        for tap in range(N_L + 1):
            H_VEL_each = np.matrix(H_VEL[:, :, tap, ch_i])
            H_VEL_ANG[:, :, tap, ch_i] = np.matmul(UR_TR, np.matmul(H_VEL_each, UT)) 
    
    
    return H_VEL_ANG # [N_RX, N_TX, (N_L + 1), N_CH/K]
