import argparse

from PIL import Image

import torch
import os
import numpy as np
import random
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from torch import nn

from crossmodelServer.HTMCModel.harnn_models.harnn import HARNN


def load_harnn():
    args = init_harnn_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 0
    net = HARNN(num_classes_list=args.num_classes_layer, total_classes=args.total_classes, vocab_size=vocab_size,
                embedding_size=args.embedding_size, lstm_hidden_size=args.lstm_hidden_size,
                attention_unit_size=args.attention_unit_size,
                fc_hidden_size=args.fc_hidden_size, beta=args.beta,
                drop_prob=args.drop_prob)
    net.eval()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(args.device)
    print("Loading state...")
    checkpoint = torch.load("crossmodelServer/HTMCModel/latest.pth", map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

def init_harnn_args():
    parser = argparse.ArgumentParser()
    #     args = parser.parse_args()
    args = parser.parse_args(args=[])
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.seed = 42
    set_seed(args.seed)
    # TODO: Change the class number of every layer, please guarantee the sum of classes per layer equals the number of total classes.
    args.num_classes_layer = [38, 648]
    args.total_classes = 686

    args.load = True

    args.print_every = 1
    args.evaluate_every = 1
    args.checkpoint_every = 10

    # TODO: Change Dimension. CURRENT: Changed. Make sure it has the same value of dimension with the output of the CN-Clip.
    args.embedding_size = 512
    args.seq_length = 256

    args.batch_size = 2
    args.epochs = 1000
    args.max_grad_norm = 0.1
    args.drop_prob = 0.5
    args.l2_reg_lambda = 0
    args.learning_rate = 5e-5
    args.beta = 0.3

    args.lstm_hidden_size = 256
    args.fc_hidden_size = 256

    args.attention_unit_size = 100

    args.threshold = 0.5
    args.top_num = 2
    args.best_auprc = 0
    return args


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = False

print("Loading CN-Clip...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./HTMCModel')
clip_model.eval()

print("Loading HARNN...")
harnn_model = load_harnn()


def get_clip_model():
    return clip_model, preprocess


def get_harnn_model():
    return harnn_model