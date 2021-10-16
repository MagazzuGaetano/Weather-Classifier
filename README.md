# Weather-Classifier
An Image Classifier for fog/rain/snow/no weather degradation implemented in Pytorch

# Experiment Hyperparameters

MEAN_STD = (          
    [0.4895458993165764, 0.4818306551357414, 0.47606749903585005],      
    [0.2629639647390231, 0.2631807514426538, 0.2756703080034581]    
)   
(Obtained by average the mean_std per each one of the classes)
    

TRAIN_SIZE = (256, 256)

TRAIN_BATCH_SIZE = 32

VAL_BATCH_SIZE = 32

MAX_EPOCH = 30

LR = 0.001

MOMENTUM = 0.9

# 

train_time: 11034015.0ms ~ 3hs

#

# Results (on test set):

Corrected Classified: 1141 / 1533

F1: 0.7442922374429224

F1 (unbalanced): 0.7521144229981033     
(Im keeping this metrics for future training because the total dataset is unbalanced, but i've trained on a balanced one)

F1 (per class): [0.83991385 0.70811744 0.57844991 0.70088496]   
(Difficulty in recognizing the class Rain)
