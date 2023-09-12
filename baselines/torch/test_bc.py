import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 
from utils import onehot
from bc_utils import get_file_names, sample_batch, read_one_file, convert_to_batch
from model import Model
from wrappers import BCWrapper

from joblib import Parallel, delayed

human_data_dir = "/home/DunkCityDynasty/human_data/L33_RELEASE"
TOTAL_DIRS = [
    "DATA_RELEASE_0",
    # "DATA_RELEASE_1",
    # "DATA_RELEASE_2",
    # "DATA_RELEASE_3",
    # "DATA_RELEASE_4",
    # "DATA_RELEASE_5",
    # "DATA_RELEASE_6",
    # "DATA_RELEASE_7",
    # "DATA_RELEASE_8",
    # "DATA_RELEASE_9",
]
file_pointers = []

import json
with open(os.path.join(f"{human_data_dir}/{TOTAL_DIRS[0]}",'file_index.json'), 'r') as f:
    data = json.load(f)
    
games = dict()
from glob import glob
for player_id in data.keys():
    games[player_id] = []
    for game_name in data[player_id].keys():
        player_one_game = glob(f"{human_data_dir}/{TOTAL_DIRS[0]}" + f'/{player_id}/*{game_name}*.jsonl')
        sorted(player_one_game, key=lambda x: int(x.split('_')[-2]))
        games[player_id].append(player_one_game)
        
print(games)