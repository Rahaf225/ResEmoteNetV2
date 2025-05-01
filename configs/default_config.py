import os

class Config:
    # Data configuration
    DATA_ROOT = 'FER2013'
    BATCH_SIZE = 64
    CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Training configuration
    EPOCHS = 100
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 20
    SEED = 42
    
    # Path configuration
    OUTPUT_DIR = 'outputs'
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
    
config = Config()