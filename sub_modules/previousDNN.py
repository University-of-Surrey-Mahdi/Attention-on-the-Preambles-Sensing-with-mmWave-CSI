# ML model for counting number of people present in the complete grid

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, concatenate, BatchNormalization

N_L             = 44        # Tap length to represent path delay
N_RX            = 4         # Number of antennas at RX device (4)
N_TX            = 4         # Number of antennas at TX device (4)
N_CH            = 128       # Number of snap shots (128)


#function Counting : Function to test localization model
# Where: 
#   Output:
#        model: Trained weights for localization ML model
def CountingPeople(MEAN_K):
    
    input1 = Input(shape=(N_RX, N_TX, (N_L + 1), int(N_CH/MEAN_K)))

    x=Flatten()(input1)
    x=BatchNormalization()(x)
    x=Dense(2000, activation='relu')(x)
    x=Dense(500,activation='relu')(x)
    output=Dense(8,activation='softmax')(x)

    model = Model(inputs=input1, outputs=output)
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.00005, decay_steps=10000, decay_rate=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    
    return model

#function Counting : Function to test localization model
# Where: 
#   Output:
#        model: Trained weights for people counting ML model
def load_NNweights(MEAN_K):
    
    # Input to the model
    input1 = Input(shape=(N_RX, N_TX, (N_L + 1), int(N_CH/MEAN_K)))
    
    # Neural Network model used for training and testing purpose
    x=Flatten()(input1)
    x=BatchNormalization()(x)
    x=Dense(2000, activation='relu')(x)
    x=Dense(500,activation='relu')(x)
    output = Dense(4, activation='softmax')(x)
    for i in range(8):
        a = Dense(4, activation='softmax')(x)
        output=concatenate([output,a])

    model = Model(inputs=input1, outputs=output)
    
    #Model parameters
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0005,
        decay_steps=10000,
        decay_rate=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    #Model compilation
    model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    
    return model

#function Testing : Function to test localization model
# Where:
#    Input : 
#       Y_Pred: prediction vector (softmax output for all the sectors for different number of people) from the ML model 
#       num_labels: maximum number of people assumed in a sector for ml model training : 4 classes : either 0 person or 1  or 2 or 3 persons 
#   Output:
#        Sector Num: number of people present in each sector for each sample
def Testing(Y_pred, num_labels):
    
    # People count in each sector of the grid
    SectorNum = np.zeros((len(Y_pred), 9))
    for i in range(len(Y_pred)):
        for j in range(9):
            SectorNum[i][j]=(np.argmax(Y_pred[i][j*num_labels:(j+1)*num_labels]))
    
    return SectorNum

#function sector_form : Funtion to convert model predictions into sector format string
# Where:
#    Input :
#       h_rad: data samples after preprocessing
#       Y_Pred: prediction vector (softmax output for all the sectors for different number of people) from the ML model 
#       Y_pred_label: prediction class labels from ML model for localization
#       SNR: snr value
#   Output: It creates a .txt file to store sector format strings
def sector_form(h_rad, Y_pred, Y_pred_label, len_pred):
    
    num_test = np.shape(Y_pred)[0]
    len_predict = np.zeros((num_test, 1))
    ind = np.zeros((num_test, 18), dtype=int)
    diff = np.zeros((num_test, 9), dtype=np.float32)
    ind_sort = np.zeros((num_test, 9), dtype=int)
    
    # output 
    len_pred_tr = np.zeros(len_pred.shape, dtype=np.float32)
    for i in range(num_test):
        len_pred_tr[i,int(np.argmax(len_pred[i,:]))] = 1
    Y_pred_tr   = np.zeros((num_test,9), dtype=int)
    lin_tr = []
    
    
    for i in range(num_test):
        a = int(np.argmax(len_pred[i,:]))
        len_predict[i] = a + 1
    # print(len_predict.shape)
    
    for i in range(num_test):
        for j in range(9):
            ind[i][2*j:2*(j+1)] = np.argpartition(Y_pred[i][j*4:(j+1)*4], -2)[-2:]
            diff[i][j] = abs(Y_pred[i][ind[i][2*j]] - Y_pred[i][ind[i][2*j + 1]])
        ind_sort[i][:] = np.argsort(diff[i][:])
    
    for i in range(num_test):
        lin = ""
        for j in range(9):
            num_people = int(np.argmax(Y_pred[i][j*4:(j+1)*4]))
            c = chr(ord('A') + j)
            for k in range(num_people):
                lin = lin + c

        p = 0
        if(len(lin) < len_predict[i]):
            aa = len_predict[i][0] - len(lin)
            for mn in range(9):
                if(aa==0):
                    break
                Sno = ind_sort[i,mn]
                curr_sector = chr(ord('A') + Sno)
                if(Y_pred[i][ind[i][2*Sno]] >= Y_pred[i][ind[i][2*Sno + 1]]):
                    if(ind[i][2*Sno+1] > ind[i][2*Sno]):
                        p = ind[i][2*Sno+1] - ind[i][2*Sno]      
                else:
                    if(ind[i][2*Sno+1] < ind[i][2*Sno]):
                        p = ind[i][2*Sno] - ind[i][2*Sno+1]
                v = int(np.min((aa,p)))
                for kk in range(v):
                    lin = lin + curr_sector
                    aa = aa - 1

            for k in range(int(aa)):
                lin = lin + 'A'

        elif(len(lin) > len_predict[i]):
            aa = len(lin) - len_predict[i][0]
            for mn in range(9):
                if(aa==0):
                    break
                Sno = ind_sort[i,mn]
                curr_sector = chr(ord('A') + Sno)
                if(Y_pred[i][ind[i][2*Sno]] >= Y_pred[i][ind[i][2*Sno + 1]]):
                    if(ind[i][2*Sno+1] < ind[i][2*Sno]):
                        p = ind[i][2*Sno] - ind[i][2*Sno+1]      
                else:
                    if(ind[i][2*Sno+1] > ind[i][2*Sno]):
                        p = ind[i][2*Sno+1] - ind[i][2*Sno]
                v = int(np.min((aa,p)))
                for kk in range(v):
                    ii = lin.find(curr_sector)
                    if(ii == -1):
                        break
                    lin_1 = lin[:ii]
                    lin_2 = lin[(ii+1):]
                    lin = lin_1 + lin_2
                    aa = aa - 1

            for k in range(int(aa)):
                lin = lin[:-1]
        
        lin_tr.append(lin)
        
        for item_i in lin:
            if item_i == "A":
                Y_pred_tr[i, 0] += 1
            elif  item_i ==  "B":
                Y_pred_tr[i, 1] += 1
            elif  item_i ==  "C":
                Y_pred_tr[i, 2] += 1
            elif  item_i ==  "D":
                Y_pred_tr[i, 3] += 1
            elif  item_i ==  "E":
                Y_pred_tr[i, 4] += 1
            elif  item_i ==  "F":
                Y_pred_tr[i, 5] += 1
            elif  item_i ==  "G":
                Y_pred_tr[i, 6] += 1
            elif  item_i ==  "H":
                Y_pred_tr[i, 7] += 1
            elif  item_i ==  "I":
                Y_pred_tr[i, 8] += 1
        # f = open(ParentFilePath+'mlOutput/mlResult'+ str(SNR) + '.txt', 'a')
        # f.write(lin)
        # f.write("\n")
        # f.close()
    return len_pred_tr, Y_pred_tr, lin_tr