# Attention on the Preamble: Indoor  Multi-Person Sensing with IEEE 802.11ay Signals
The framework for multi-object sensing with communication-centric signals and thus for solving the problem stetement [ITU-ML5G-PS-002: WALDO (Wireless Artificial intelligence Location DetectiOn): sensing using mmWave communications and ML](https://challenge.aiforgood.itu.int/match/matchitem/38/) [1] provided by the ITU AI/ML in 5G Challenge 2021 in collaboration with NIST. Below figure shows an overview of the problem statement. We would like to count the persons in the room and localize/classify them into 9 sectors (A to I) accurately.\
![Problem statement](https://github.com/University-of-Surrey-Mahdi/WiFi-sensing/assets/124618252/0f8a18f8-ab65-43bf-a3d8-140da052e2fb)

Below figure shows an overall block diagram of the proposed framework which is similar to a previous study [2]. We apply vision transformer (ViT) [3] in the final solution and design the output layer as predicting probability density of human presence in each sector. Furthermore, we pre-process received signals as beams are adjusted for indoor sensing, and apply curriculum learning [4] known to be effective for gradually more complex tasks to train ViT as predicting various SNRs data.\
Our numerical results show 79.4\% counting accuracy and 39.7\% localization one when SNR is 18dB.
![Proposed framework](https://github.com/University-of-Surrey-Mahdi/WiFi-sensing/assets/124618252/53f4d351-20c2-46db-8ff8-719e0c7a85c6)

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
| `sub_modules` | `/pre_doppler.py` | Code | Pre-process | Doppler domain analysis. Estimated channels for 128 consecutive packets are transfered to estimated channels for each velocity $(\Rightarrow \hat{\boldsymbol{H}}_{\textrm{vel}})$.|
| `sub_modules` | `/pre_angle.py` | Code | Pre-process | Angular domain analysis. Estimated channels for 4Tx $\times$ 4Rx antennas are transfered to estimated channels for each angle $(\Rightarrow \hat{\boldsymbol{H}}_{\textrm{vel}}^{\textrm{a}})$.|
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
2) S. Khunteta, A. K. R. Chavva, and A. Agrawal, "AI-based indoor localization using mmWave MIMO channel at 60 GHz," ITU Journal on Future and Evolving Technologies, vol. 3, no. 2, pp. 243-251, 2022.
3) A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," 2021.
4) Y. Bengio, J. Louradour, R. Collobert, and J. Weston, "Curriculum Learning," Int. Conf. Mach. Learn., pp. 41-48, 2009.
5) C. Gustafson, K. Haneda, S. Wyne, and F. Tufvesson, "On mm-Wave Multipath Clustering and Channel Modeling," IEEE Transactions on Antennas and Propagation, vol. 62, no. 3, pp. 1445-1455, 2014.
6) P. Hanpinitsak, K. Saito, J.-i. Takada, M. Kim, and L. Materum, "Multipath Clustering and Cluster Tracking for Geometry-Based Stochastic Channel Modeling," IEEE Transactions on Antennas and Propagation, vol. 65, no. 11, pp. 6015-6028, 2017.


# Publication
- Tatsuya Kikuzuki, Mahdi Boloursaz Mashhadi, Yi Ma, and Rahim Tafazolli, "Attention on Preambles: Indoor Multi-Object Sensing with IEEE 802.11ay Networks," IEEE Wireless Communication Letters, in press.

# Contacts
- Tatsuya Kikuzuki (FUJITSU LIMITED): kikuzuki{at}fujitsu.com
- Mahdi Boloursaz Mashhadi (The University of Surrey): m.boloursazmashhadi{at}surrey.ac.uk
