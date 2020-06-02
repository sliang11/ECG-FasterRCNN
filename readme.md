# Overview

This repository holds the source code of our PAKDD 2020 paper "Simultaneous ECG Heartbeat Segmentation and Classification with Feature Fusion and Long Term Context Dependencies". Please note that some of the code is transferred from other repositories. Please use our code, as well as that shared by other contributors, in accordance with the respective licenses and protocols.
 
# Usage
Please take the following steps to run our code.

 1. Download the MIT-BIH database to directory:/data_2 
 2. cd tool/  
 3. Run in shell/terminal: python setup.py build develop  
 4. cd tool/batch  
 5. Run in shell/terminal: python setup.py build develop  
 6. Run cls1.py or cls2.py to select a pre-trained model and rename as  "base_max.p" 
 7. Run train_test1.py to get segment result and model, rename as 'base_a1_max.p' and 'rpn_a1_max.p'
 8. Run train_test2.py to get classification result 
 
 # Environment
 Our experiments were run in the following environment.
 
 Python 3.7  
 Pytorch 1.1.0  
 Cuda 10.0  
 Ubuntu 18.04.1
