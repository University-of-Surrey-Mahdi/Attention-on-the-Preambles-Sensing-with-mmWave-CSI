# Attention on the Preamble: Indoor  Multi-Person Sensing with IEEE 802.11ay Signals
The ubiquitous availability of wireless networks and devices provides a unique opportunity to leverage the corresponding communication-centric signals to enable wireless sensing applications. In this article, we propose a new framework for bi-static indoor multi-person counting and localization tasks utilizing the IEEE 802.11ay signals (refer to [1]). In the proposed framework, multi-modal delay, Doppler, angular features are first derived from the estimated Channel State Information (CSI) using the IEEE 802.11ay preamble signals at the receiver, and then a modified attention-based Vision Transformer (ViT) [2] architecture is designed to extract the most effective features for the counting and localization tasks. We also develop a novel post-processing technique based on combinatorial Kullback-Leibler (KL) minimization to transfer knowledge from counting to the localization task, thereby significantly simplifying the Neural Network (NN) architecture. Our numerical results show accurate counting and localization capabilities that significantly outperform the existing works based on pure conventional signal processing techniques [3], or the state-of-the-art ML-based approaches [4].

# How to use the code

1. Download the dataset via [ML5G-PS-002](https://challenge.aiforgood.itu.int/match/matchitem/38/) [1]
2. Change the path of dataset in `MAIN.py` (e.g., `DATA_PATH`)
3. Install python and required packages. It is highly required to construct CUDA environment because ViT takes time to train/predict.
4. Run the code `MAIN.py` on python environment

Note that you can set the RUN MODE if you want to skip some processes
- `SOLUTION_MODE`: Switch whether you run ViT/CNN/DNN/geometry-based process
- `PREPRO_MODE`: Switch whether pre-process runs
- `PREPRO_SAVE_MODE`: Switch whether pre-processed data will be stored
- `FULL_DATA_MODE`: Switch whether program load all data
- `TRAIN_MODE`: Switch whether ML models will be trained

![Flow](https://github.com/University-of-Surrey-Mahdi/WiFi-sensing/assets/124618252/a5331c69-18ff-491d-91a0-97e471f63ed7)


# Repository Overview
| Name | (Sub name) | Category | (Sub category) | Description |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| `MAIN.py` || Code | Main | A main python code for running "Attention on Preambles: Indoor Multi-Object Sensing with IEEE 802.11ay Networks" |
| `sub_modules` | `/load_gt.py` | Code | Read | Read the ground truths both of counting and of localization |
| `sub_modules` | `/pre_chest.py` | Code | Pre-process | CH estimation. Raw received signals $(\boldsymbol{Y})$ are transfered to estimated channels $(\Rightarrow \hat{\boldsymbol{H}})$ using known transmitted signal $(\boldsymbol{X})$. |
| `sub_modules` | `/tx.mat` | Code | Pre-process | Known transmitted signal $(\boldsymbol{X})$ |
| `sub_modules` | `/pre_doppler.py` | Code | Pre-process | Doppler domain analysis. Estimated channels for 128 consecutive packets are transfered to estimated channels for each velocity $(\Rightarrow \hat{\boldsymbol{H}}_{\textrm{Dopp}})$.|
| `sub_modules` | `/pre_angle.py` | Code | Pre-process | Angular domain analysis. Estimated channels for 4Tx $\times$ 4Rx antennas are transfered to estimated channels for each angle $(\Rightarrow \hat{\boldsymbol{H}}_{\textrm{Dopp}}^{\textrm{Ang}})$.|
| `sub_modules` | `/ml_models.py` | Code | ML | Proposed ML models (CNN and ViT). Output the initial model, train the model, or predict the result with the model. |
| `sub_modules` | `/previousDNN.py` | Code | ML | Previous ML model (DNN). Output the initial model, or counting/localization result (including post-process). Note that training the model is not included because it is built in keras library.|
| `sub_modules` | `/previousGEO.py` | Code | ML | Previous geometry-based algorithm and ML model (SVM). Output the MPC (multi-path components), or counting/localization result (including post-process). |
| `sub_modules` | `/post.py` | Code | Post-process | Proposed post-process algorithm. The predicted results with ML models (CNN and ViT) are transfered to counting/localization result. |
| `sub_modules` | `/report.py` | Code | Report | Evaluate, save, and print the counting/localization result. |
| `trained_model` || Model  || Pre-trained models are stored |
| `model` || Model  || Trained models while running the code are stored. This directory is vacant as a default, but it is possible to run the code with test mode if you copy the models stored in `trained_model` here. |
| `output` || Output  || The output results (.txt) are stored with the format of [ML5G-PS-002](https://challenge.aiforgood.itu.int/match/matchitem/38/) |
| `README.md` || README  || You are here! |
| `Spec.pptx` || README  || More detailed explanation of source code |


# Required python packages
Please install packages below before running `MAIN.py`
- os
- gc
- copy
- itertools
- pickle
- numpy
- scipy
- sklearn
- torch
- tensorflow

# References
1) [ML5G-PS-002](https://challenge.aiforgood.itu.int/match/matchitem/38/)
2) A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," 2021.
3) P. Hanpinitsak, K. Saito, J.-i. Takada, M. Kim, and L. Materum, "Multipath Clustering and Cluster Tracking for Geometry-Based Stochastic Channel Modeling," IEEE Transactions on Antennas and Propagation, vol. 65, no. 11, pp. 6015-6028, 2017.
4)  S. Khunteta, A. K. R. Chavva, and A. Agrawal, "AI-based indoor localization using mmWave MIMO channel at 60 GHz," ITU Journal on Future and Evolving Technologies, vol. 3, no. 2, pp. 243-251, 2022. 


# Publication
- Tatsuya Kikuzuki, Mahdi Boloursaz Mashhadi, Yi Ma, and Rahim Tafazolli, "Attention on the Preamble: Indoor  Multi-Person Sensing with IEEE 802.11ay Signals," submitted for publication. 

# Contacts
- Tatsuya Kikuzuki (FUJITSU LIMITED): kikuzuki{at}fujitsu.com
- Dr. Mahdi Boloursaz Mashhadi (The University of Surrey): m.boloursazmashhadi{at}surrey.ac.uk
