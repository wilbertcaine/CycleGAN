import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_X = "datasets/horse2zebra/trainA"
TRAIN_DIR_Y = "datasets/horse2zebra/trainB"
VAL_DIR_X = "datasets/horse2zebra/testA"
VAL_DIR_Y = "datasets/horse2zebra/testB"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0.5 * LAMBDA_CYCLE
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = True
NAME = "CycleGAN"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
