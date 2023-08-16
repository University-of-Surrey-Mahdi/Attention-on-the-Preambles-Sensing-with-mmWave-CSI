# Attention on Preambles: Indoor Multi-Object Sensing with IEEE 802.11ay Networks
The framework for multi-object sensing with communication-centric signals and thus for solving the problem stetement [ITU-ML5G-PS-002: WALDO (Wireless Artificial intelligence Location DetectiOn): sensing using mmWave communications and ML](https://challenge.aiforgood.itu.int/match/matchitem/38/) [1] provided by the ITU AI/ML in 5G Challenge 2021 in collaboration with NIST. Below figure shows an overview of the problem statement. We would like to count the persons in the room and localize/classify them into 9 sectors (A to I) accurately.\
![Problem statement](https://github.com/University-of-Surrey-Mahdi/WiFi-sensing/files/12359954/pdfresizer.com-pdf-crop.pdf)

Below figure shows an overall block diagram of the proposed framework which is similar to a previous study [2]. We apply vision transformer (ViT) [3] in the final solution and design the output layer as predicting probability density of human presence in each sector. Furthermore, we pre-process received signals as beams are adjusted for indoor sensing, and apply curriculum learning [4] known to be effective for gradually more complex tasks to train ViT as predicting various SNRs data.\
![Proposed framework](https://github.com/University-of-Surrey-Mahdi/WiFi-sensing/files/12359985/pdfresizer.com-pdf-crop.1.pdf)\
Below figure explains the design of the output layer as predicting probability density of human presence in each sector.\
![Output neuron design](https://github.com/University-of-Surrey-Mahdi/WiFi-sensing/files/12359994/pdfresizer.com-pdf-crop.2.pdf)

Our numerical results show 79.4\% counting accuracy and 39.7\% localization one when SNR is 18dB. Below are detailed descriptions of results.\
The counting result (see figure (a)) shows the advantage of our proposed framework with ViT which improves the counting accuracy in comparison with the benchmark solutions when SNR is high (20.9\% of DNN-based [2] or 39.2\% of geometry-based [5][6] to 79.4\% when SNR is 18 dB).\
The localization result (see figure (b)) shows the advantage of our proposed framework with ViT which improves the localization accuracy in comparison with the benchmark solutions especially when SNR is high (2.2\% of DNN-based [2] or 2.7\% of geometry-based [5][6] to 39.7\% when SNR is 18 dB).\
Figure (c) shows the detailed localization accuracy in each number of persons when SNR is 18 dB. Our proposed framework with ViT achieves more than 76\% or 41\% localization accuracy when number of persons is less than or equals to 3 or 6.\
![Numerical results](https://github.com/University-of-Surrey-Mahdi/WiFi-sensing/files/12360006/pdfresizer.com-pdf-crop.3.pdf)

# How to use the code

1. Download the dataset via [ML5G-PS-002](https://challenge.aiforgood.itu.int/match/matchitem/38/) [1]
2. Change the path of dataset in "MAIN.py" (e.g., DATA_PATH)
3. Install python and required packages. It is highly required to construct CUDA environment because ViT takes time to train/predict.
4. Run the code "MAIN.py" on python environment

Note that you can set the RUN MODE if you want to skip some processes
- SOLUTION_MODE: Switch whether you run ViT/CNN/DNN/geometry-based process
- PREPRO_MODE: Switch whether pre-process runs
- PREPRO_SAVE_MODE: Switch whether pre-processed data will be stored
- FULL_DATA_MODE: Switch whether program load all data
- TRAIN_MODE: Switch whether ML models will be trained

![Flow](https://github.com/University-of-Surrey-Mahdi/WiFi-sensing/files/12360034/pdfresizer.com-pdf-crop.4.pdf)


# Repository Overview
`MAIN.py`: A main python code for running "Attention on Preambles: Indoor Multi-Object Sensing with IEEE 802.11ay Networks"

`sub_modules/XXX`: Sub modules whose functions are called by `MAIN.py`

`trained_model/XXX`: Pre-trained models

`model/XXX`: Temporary directory for storing the trained models when running the code. If you copy the models stored in `trained_model` here, it is possible to run the code with test mode.

`output/XXX`: Temporary directory for storing the output results with the format of  [ML5G-PS-002](https://challenge.aiforgood.itu.int/match/matchitem/38/)

`Code_specification.pptx`: Explanation of Source code

`README.md`: You are here!

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
