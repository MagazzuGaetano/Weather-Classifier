DATA_PATH = './preprocessed_data'
MEAN_STD = (
    [0.4895458993165764, 0.4818306551357414, 0.47606749903585005], 
    [0.2629639647390231, 0.2631807514426538, 0.2756703080034581]
)

SEED = 314
PRETRAINED = False
RESUME = False
RESUME_PATH = './latest_state.pth'

TRAIN_SIZE = (256, 256) #(224, 224)
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16

MAX_EPOCH = 30 #70
LR = 1e-5 if PRETRAINED else 0.001 # there still a problem with transfer learnig!
MOMENTUM = 0.9

CLASSES = ['no weather degradation', 'fog', 'rain', 'snow']
NUM_CLASSES = len(CLASSES)

