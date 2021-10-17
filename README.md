# Weather-Classifier
An Image Classifier for fog/rain/snow/no weather degradation implemented in Pytorch, the model selected for this study is the ResNet50.     
The dataset is a combination of the [JHU-CROWD++](http://www.crowd-counting.com/) and the [Rain Fog Snow Dataset](https://github.com/ZebaKhanam91/SP-Weather).

# Experiment Hyperparameters

MEAN = [0.4895458993165764, 0.4818306551357414, 0.47606749903585005]    
STD = [0.2629639647390231, 0.2631807514426538, 0.2756703080034581]     
(Obtained by average the mean_std per each one of the classes)
    
PRETRAINED=False # Transfer learning Bad Results (I have to freeze less layers)     

TRAIN_SIZE = (256, 256)

TRAIN_BATCH_SIZE = 16

VAL_BATCH_SIZE = 16

LR = 0.001

MOMENTUM = 0.9

# Results (on test set):

train_time: 2017029.0ms ~ 33 mins 70 epochs     
MAX_EPOCH: 70   
corrected classified: 1260 / 1533   
F1: 0.821917808219178   
F1 (per class): [0.8766368  0.82226981 0.70187394 0.8057041]    
F1 (unbalanced): 0.8272478222940107     


train_time: 4333681.5ms ~ 1,2 hs 150 epochs      
MAX_EPOCH: 150      
corrected classified: 1287 / 1533       
F1: 0.8395303326810175      
F1 (per class): [0.89566396 0.84599589 0.73511294 0.78246753]       
F1 (unbalanced): 0.842190835711799  

Difficulty in recognize the Rain class      
F1 (unbalanced) Im keeping this metrics for future training because the total dataset is unbalanced, but i've trained on a balanced one


# TO DO:

- Investigate on the Rain samples!

- Transfer Learning from ImageNet (To Fix)


# Citations:

- (JHU-CROWD++) Sindagi, Vishwanath A and Yasarla, Rajeev and Patel, Vishal M, [JHU-CROWD++: Large-Scale Crowd Counting Dataset and A Benchmark Method](https://arxiv.org/abs/2004.03597), IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), (2020)

- (RFS Dataset) Jose Carlos Villarreal Guerra, Zeba Khanam, Shoaib Ehsan, Rustam Stolkin, Klaus McDonald-Maier, [Weather Classification: A new multi-class dataset, data augmentation approach and comprehensive evaluations of Convolutional Neural Networks](https://arxiv.org/pdf/1808.00588.pdf), (2018) 
