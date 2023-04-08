

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report
import preprocessdata as predata
import training as train


# Reading the dataset
dataset_path = "/content/drive/MyDrive/NLPassignment/Question2/Dataset/"

train.main(dataset_path)