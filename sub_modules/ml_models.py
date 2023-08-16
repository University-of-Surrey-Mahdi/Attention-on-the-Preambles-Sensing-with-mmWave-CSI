
### CNN and ViT models

# Source reference for ViT
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html

# Standard libraries
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
# Special libraries
import sub_modules.post as post
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Fixed parameters
N_L             = 44
N_RX            = 4
N_TX            = 4
N_CH            = 128

# Hyper parameters for ViT
EMBED_DIM       = 256
HIDDEN_DIM      = 512
NUM_HEADS       = 8
NUM_LAYERS      = 6
PATCH_SIZE1     = 3
PATCH_SIZE2     = 2
DROPOUT         = 0.2

# Hyper parameters for CNN
INPUT_CHANNELS  = 64
LAYER1_NEURONS  = 128
LAYER2_NEURONS  = 32
ACTIVATION      = torch.nn.ReLU()

# Variable parameters
BATCH_SIZE      = 16
NUM_ALL         = 8 # 1
NUM_NEU         = 9 # 8


def img_to_patch(x, patch_size1, patch_size2, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size1, patch_size1, W//patch_size2, patch_size2)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x


class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, model_kwargs):
    # def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size1, patch_size2, num_patches, MEAN_K, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()
        self.institute(**model_kwargs)
    
    def institute(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size1, patch_size2, num_patches, MEAN_K, dropout=0.0):
    
        self.patch_size1 = patch_size1
        self.patch_size2 = patch_size2

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size1*patch_size2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))


    def forward(self, x):
        # Preprocess input
        x = x.reshape(x.shape[0], 4 * 4, x.shape[3], x.shape[4]) # -, 16[channels], 45[taps], 128/MEAN_K[vel]
        x = img_to_patch(x, self.patch_size1, self.patch_size2)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out


class CNN(torch.nn.Module):
    def __init__(self, MEAN_K, OUTPUT_NEURONS):
        super(CNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(4 * 4, 32, (6, int(32/MEAN_K+1)), stride=(1, 1))
        self.pool1 = torch.nn.MaxPool2d(2, 2)    # pooling (2x2)
        self.conv2 = torch.nn.Conv2d(32, INPUT_CHANNELS, (3, int(16/MEAN_K+1)), stride=(1, 1))
        self.pool2 = torch.nn.MaxPool2d(2, 2)    # pooling (2x2)
        
        self.bn0        = torch.nn.BatchNorm1d(INPUT_CHANNELS * 9 * int(16/MEAN_K))
        self.layer1     = torch.nn.Linear(INPUT_CHANNELS * 9 * int(16/MEAN_K), LAYER1_NEURONS)
        self.bn1        = torch.nn.BatchNorm1d(LAYER1_NEURONS)
        self.layer2     = torch.nn.Linear(LAYER1_NEURONS, LAYER2_NEURONS)
        self.bn2        = torch.nn.BatchNorm1d(LAYER2_NEURONS)
        self.layer_out  = torch.nn.Linear(LAYER2_NEURONS, OUTPUT_NEURONS)
        
    def forward(self, x):
        
        x = x.reshape(x.shape[0], 4 * 4, x.shape[3], x.shape[4]) # -, 16[channels], 45[taps], 128/MEAN_K[vel]
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool2(x)
        
        x = x.reshape(x.shape[0], -1)
        x = ACTIVATION(self.layer1(x))
        x = self.bn1(x)
        x = ACTIVATION(self.layer2(x))
        x = self.bn2(x)
        x = self.layer_out(x)
        
        return x


def init_NN(MEAN_K, num_all, num_neu, model_type):
    
    global NUM_ALL
    global NUM_NEU
    
    NUM_ALL     = num_all
    NUM_NEU     = num_neu
    
    if model_type == "ViT":
        model = VisionTransformer(model_kwargs={
                                        'embed_dim': EMBED_DIM,
                                        'hidden_dim': HIDDEN_DIM,
                                        'num_heads': NUM_HEADS,
                                        'num_layers': NUM_LAYERS,
                                        'patch_size1': PATCH_SIZE1,
                                        'patch_size2': PATCH_SIZE2,
                                        'num_channels': N_RX * N_TX,
                                        'num_patches': int(((N_L + 1) / PATCH_SIZE1) * (N_CH / MEAN_K / PATCH_SIZE2)),
                                        'num_classes': NUM_NEU,
                                        'MEAN_K': MEAN_K,
                                        'dropout': DROPOUT
                                    })
    elif model_type == "CNN":
        model = CNN(MEAN_K, NUM_NEU)
    else:
        print("NO model type of " + model_type + "!")
    
    return model.to(device)



def train_NN(model, h_rad_train, count_array_train, gt_array_train, h_rad_val, count_array_val, gt_array_val, N_EPOCH, LEARNING_RATE, WEIGHT_DECAY, batch_size):

    post.set_param(NUM_ALL, NUM_NEU)
    global BATCH_SIZE
    BATCH_SIZE          = batch_size
    
    # transfer
    count_array_train   = np.argmax(count_array_train, axis=1) + 1
    count_array_val     = np.argmax(count_array_val, axis=1) + 1
    num_all_train       = gt_array_train.sum(axis=1)
    num_all_val         = gt_array_val.sum(axis=1)
    gt_arrayPD_train    = np.zeros(gt_array_train.shape)
    gt_arrayPD_val      = np.zeros(gt_array_val.shape)
    
    for train_i in range(int(len(gt_array_train))):
        gt_arrayPD_train[train_i] = gt_array_train[train_i] / gt_array_train[train_i].sum()
    for val_i in range(int(len(gt_array_val))):
        gt_arrayPD_val[val_i] = gt_array_val[val_i] / gt_array_val[val_i].sum()
        
    # Transform to torch format
    t_X_train_4D    = torch.from_numpy(h_rad_train.astype(np.float32))
    t_y_train       = torch.from_numpy(np.vstack((gt_array_train, gt_array_train, gt_array_train)).astype(np.float32))
    t_yPD_train     = torch.from_numpy(np.vstack((gt_arrayPD_train, gt_arrayPD_train, gt_arrayPD_train)).astype(np.float32))
    t_c_train       = torch.from_numpy(np.hstack((num_all_train, num_all_train, num_all_train)).astype(np.float32))
    t_count_train   = torch.from_numpy(np.hstack((count_array_train, count_array_train, count_array_train)).astype(np.float32))
    
    data_len_each   = int(len(gt_array_val))
    t_X_valid_4D    = [[0 for j in range(3)] for i in range(8)]
    t_y_valid       = [[0 for j in range(3)] for i in range(8)]
    t_yPD_valid     = [[0 for j in range(3)] for i in range(8)]
    t_c_valid       = [[0 for j in range(3)] for i in range(8)]
    t_count_valid   = [[0 for j in range(3)] for i in range(8)]
    
    for i in range(8):
        for j in range(3):
            valid_index = (np.hstack((count_array_val, count_array_val, count_array_val)) == (i + 1)) & (np.array(range(int(len(h_rad_val)))) >= data_len_each*j) & (np.array(range(int(len(h_rad_val)))) < data_len_each*(j+1))
            t_X_valid_4D[i][j]  = torch.from_numpy(h_rad_val[valid_index].astype(np.float32))
            
            if j == 0:
                t_y_valid[i][0]     = torch.from_numpy(np.vstack((gt_array_val, gt_array_val, gt_array_val))[valid_index].astype(np.float32))
                t_y_valid[i][1]     = torch.from_numpy(np.vstack((gt_array_val, gt_array_val, gt_array_val))[valid_index].astype(np.float32))
                t_y_valid[i][2]     = torch.from_numpy(np.vstack((gt_array_val, gt_array_val, gt_array_val))[valid_index].astype(np.float32))
                t_yPD_valid[i][0]   = torch.from_numpy(np.vstack((gt_arrayPD_val, gt_arrayPD_val, gt_arrayPD_val))[valid_index].astype(np.float32))
                t_yPD_valid[i][1]   = torch.from_numpy(np.vstack((gt_arrayPD_val, gt_arrayPD_val, gt_arrayPD_val))[valid_index].astype(np.float32))
                t_yPD_valid[i][2]   = torch.from_numpy(np.vstack((gt_arrayPD_val, gt_arrayPD_val, gt_arrayPD_val))[valid_index].astype(np.float32))
                t_c_valid[i][0]     = torch.from_numpy(np.hstack((num_all_val, num_all_val, num_all_val))[valid_index].astype(np.float32))
                t_c_valid[i][1]     = torch.from_numpy(np.hstack((num_all_val, num_all_val, num_all_val))[valid_index].astype(np.float32))
                t_c_valid[i][2]     = torch.from_numpy(np.hstack((num_all_val, num_all_val, num_all_val))[valid_index].astype(np.float32))
                t_count_valid[i][0] = torch.from_numpy(np.hstack((count_array_val, count_array_val, count_array_val))[valid_index].astype(np.float32))
                t_count_valid[i][1] = torch.from_numpy(np.hstack((count_array_val, count_array_val, count_array_val))[valid_index].astype(np.float32))
                t_count_valid[i][2] = torch.from_numpy(np.hstack((count_array_val, count_array_val, count_array_val))[valid_index].astype(np.float32))
            
    
    del h_rad_train, count_array_train, num_all_train, gt_array_train, gt_arrayPD_train, h_rad_val, count_array_val, num_all_val, gt_array_val, gt_arrayPD_val
    
    
    # loss, weighted acc
    avg_loss        = 0.0   # average loss for training data
    avg_acc         = 0.0   # average weighted acc for training data
    avg_val_loss    = np.zeros((8, 3))   # average loss for validation data
    avg_val_acc     = np.zeros((8, 3))   # average weighted acc for validation data
    train_history       = np.zeros((N_EPOCH))
    valid_history       = np.zeros((8, 3, N_EPOCH))   
    trainacc_history    = np.zeros((N_EPOCH))   
    validacc_history    = np.zeros((8, 3, N_EPOCH))   
    
    # Prepare dataset loader
    dataset_train   = TensorDataset(t_X_train_4D, t_y_train, t_yPD_train, t_c_train, t_count_train)        # for training 
    loader_train    = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_valid_list = [[0 for j in range(3)] for i in range(8)]
    for i in range(8):
        for j in range(3):
            dataset_valid = TensorDataset(t_X_valid_4D[i][j], t_y_valid[i][j], t_yPD_valid[i][j], t_c_valid[i][j], t_count_valid[i][j])    # for validation i, j
            loader_valid  = DataLoader(dataset_valid, batch_size=BATCH_SIZE)
            loader_valid_list[i][j] = loader_valid
    
    del t_X_train_4D, t_y_train, t_yPD_train, t_c_train, t_count_train, t_X_valid_4D, t_y_valid, t_yPD_valid, t_c_valid, t_count_valid
    del dataset_train, dataset_valid, loader_valid
    
    # Define optimizer
    my_optim = torch.optim.Adam(
                                model.parameters(),         # parameters updated by optimizer (weight & bias)
                                lr=LEARNING_RATE,           # learning rate
                                weight_decay=WEIGHT_DECAY   # weight decay
                                )
    
    # Epoch loop for training
    for epoch in range(N_EPOCH):
        
        # parameters used in local loop
        total_loss      = 0.0   # cumulative loss value for training
        total_acc       = 0.0   # cumulative weighted acc for training
        total_val_loss  = np.zeros((8, 3))  # cumulative loss value for validation
        total_val_acc   = np.zeros((8, 3))   # cumulative weighted acc for validation
        total_train_Pn  = 0     # cummulative person count for training
        total_valid_Pn  = np.zeros((8, 3))     # cummulative person count for validation
        total_train_n   = 0     # cummulative count for training
        total_valid_n   = np.zeros((8, 3))    # cummulative count for validation
        
        model.train() # training mode
        for t_X_train_local, t_y_train_local, t_yPD_train_local, t_c_train_local, t_count_train_local in loader_train:
            
            t_X_train_local = t_X_train_local.to(device)
            t_y_train_local = t_y_train_local.to(device)   
            t_yPD_train_local   = t_yPD_train_local.to(device) 
            t_c_train_local     = t_c_train_local.to(device)
            t_count_train_local = t_count_train_local.to(device)

            # training for 1 mini batch
            pred_y = model(t_X_train_local)    # predicted value
            
            pred_acc = post.pred(pred_y, t_c_train_local)

            acc = torch.matmul(t_count_train_local.type(torch.FloatTensor),
                                   (pred_acc == t_y_train_local).all(1).type(torch.FloatTensor))
            acc_Pn = t_count_train_local.sum()

            # calculate loss and gradient
            # Due to the fact that loss function is not a library-based one, do not calculate backward when catching the error.
            my_optim.zero_grad()                        # initialize loss with 0
            try:
                loss = torch.mean(t_count_train_local * torch.sum(- t_yPD_train_local * torch.log((torch.nn.functional.softmax(pred_y, dim=1))) , 1) / NUM_NEU)
                loss.backward()                             # backward (Autograd)
                # update parameters (bias & weight)
                my_optim.step()                             # optimize
            
                # cummulative process
                total_loss      += loss.item()      # cumulative loss
                total_acc       += acc.item()       # cumulative acc
                total_train_Pn  += acc_Pn.item()    # cummulative person count
                total_train_n   += 1                # cummulative count
            
            except:
                print("CAUTION: skip the backward function because of loss error")
        
        model.eval() # validation mode
        with torch.no_grad():
            for count_idx, loader_valid_l in enumerate(loader_valid_list):
                for snr_idx, loader_valid in enumerate(loader_valid_l):
                
                    for t_X_valid_local, t_y_valid_local, t_yPD_valid_local, t_c_valid_local, t_count_valid_local in loader_valid:
                        
                        t_X_valid_local     = t_X_valid_local.to(device)
                        t_y_valid_local     = t_y_valid_local.to(device) 
                        t_yPD_valid_local   = t_yPD_valid_local.to(device) 
                        t_c_valid_local     = t_c_valid_local.to(device)
                        t_count_valid_local = t_count_valid_local.to(device)

                        # validation for 1 mini batch
                        pred_y = model(t_X_valid_local)    # predicted value
                        
                        pred_acc = post.pred(pred_y, t_c_valid_local)

                        val_acc = torch.matmul(t_count_valid_local.type(torch.FloatTensor),
                                           (pred_acc == t_y_valid_local).all(1).type(torch.FloatTensor))
                        val_acc_Pn = t_count_valid_local.sum()

                        # calculate loss only
                        try:
                            val_loss = torch.mean(t_count_valid_local * torch.sum(- t_yPD_valid_local * torch.log((torch.nn.functional.softmax(pred_y, dim=1))) , 1) / NUM_NEU)
                        
                            # cummulative process
                            total_val_loss[count_idx][snr_idx]     += val_loss.item()      # cumulative loss
                            total_val_acc[count_idx][snr_idx]      += val_acc.item()       # cumulative acc
                            total_valid_Pn[count_idx][snr_idx]     += val_acc_Pn.item()    # cummulative person count
                            total_valid_n[count_idx][snr_idx]      += 1                    # cummulative count
                        
                        except:
                            print("CAUTION: skip the backward function because of loss error (test)")
        
        # Calculate mean value for cumulative value with mini batch units
        avg_loss = total_loss / total_train_n           # average loss for training data
        avg_acc = total_acc / total_train_Pn            # average weighted acc for training data
        train_history[epoch] = avg_loss
        trainacc_history[epoch] = avg_acc
        for count_idx in range(8):
            for snr_idx in range(3):
                avg_val_loss[count_idx][snr_idx] = total_val_loss[count_idx][snr_idx] / total_valid_n[count_idx][snr_idx]    # average loss for validation data
                avg_val_acc[count_idx][snr_idx] = total_val_acc[count_idx][snr_idx] / total_valid_Pn[count_idx][snr_idx]     # average weighted acc for validation data
    
                valid_history[count_idx][snr_idx][epoch] = avg_val_loss[count_idx][snr_idx]
                validacc_history[count_idx][snr_idx][epoch] = avg_val_acc[count_idx][snr_idx]
                            
        # Report
        if (epoch+1) % 10 == 0:

            val_loss_report = []
            val_acc_report  = []
            for snr_idx in range(3):
                loss_local      = 0.0
                loss_local_n    = 0.0
                acc_local       = 0.0
                acc_local_Pn    = 0.0
                for count_idx in range(8):
                    loss_local      += total_val_loss[count_idx][snr_idx]
                    loss_local_n    += total_valid_n[count_idx][snr_idx]
                    acc_local       += total_val_acc[count_idx][snr_idx]
                    acc_local_Pn    += total_valid_Pn[count_idx][snr_idx]
                val_loss_report.append(loss_local / loss_local_n)
                val_acc_report.append(acc_local / acc_local_Pn)

            print(f'[Epoch {epoch+1:3d}/{N_EPOCH:3d}]'\
                  f' loss: {avg_loss:.5f}, acc: {avg_acc:.5f}')
            print(f'For test dataset' \
                  '\n'
                  f' val_loss(+18): {val_loss_report[2]:.5f}, val_acc(+18): {val_acc_report[2]:.5f}' \
                  '\n'
                  f' val_loss(  0): {val_loss_report[1]:.5f}, val_acc(  0): {val_acc_report[1]:.5f}' \
                  '\n'
                  f' val_loss(-18): {val_loss_report[0]:.5f}, val_acc(-18): {val_acc_report[0]:.5f}')
            # Detailed results
            # for count_idx in range(8):
            #     print(f'N_PERSON = {count_idx+1}' \
            #       '\n'
            #       f' val_loss(+18): {avg_val_loss[count_idx][2]:.5f}, val_acc(+18): {avg_val_acc[count_idx][2]:.5f}' \
            #       '\n'
            #       f' val_loss(  0): {avg_val_loss[count_idx][1]:.5f}, val_acc(  0): {avg_val_acc[count_idx][1]:.5f}' \
            #       '\n'
            #       f' val_loss(-18): {avg_val_loss[count_idx][0]:.5f}, val_acc(-18): {avg_val_acc[count_idx][0]:.5f}')
        
    return model, train_history, trainacc_history, valid_history, validacc_history


def predict_NN(model, h_rad, batch_size):
    global BATCH_SIZE
    BATCH_SIZE          = batch_size
    
    t_X_4D_all  = torch.from_numpy(h_rad.astype(np.float32))
    del h_rad
    model           = model.to(device)
    
    batch_count = int(len(t_X_4D_all) / BATCH_SIZE)
    pred_y_np   = np.zeros((len(t_X_4D_all), NUM_NEU), dtype=float)
    model.eval()
    with torch.no_grad():
        for i in range(batch_count):
            
            t_X_4D  = t_X_4D_all[BATCH_SIZE * i:BATCH_SIZE * (i + 1)].to(device)
            
            pred_y_np[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]      = model(t_X_4D).detach().cpu().numpy()
            
        t_X_4D  = t_X_4D_all[BATCH_SIZE * batch_count:int(len(t_X_4D_all))].to(device)
        pred_y_np[BATCH_SIZE * batch_count:int(len(t_X_4D_all))]      = model(t_X_4D).detach().cpu().numpy()
    
    
    return pred_y_np

