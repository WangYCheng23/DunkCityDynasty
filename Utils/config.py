import argparse
import os
from pathlib import Path

import torch

def readParser():
    parser = argparse.ArgumentParser(description='Dunk_city_dynasty')
    
    # Env Configuration
    parser.add_argument('--env_setting',default='linux')
    parser.add_argument('--client_path',default=os.path.join(Path(__file__).parent.parent.resolve(), 'game_package_release'))
    parser.add_argument('--rl_server_ip',default='127.0.0.1')
    parser.add_argument('--rl_server_port',default=42636)
    parser.add_argument('--game_server_ip',default='121.40.214.152')
    parser.add_argument('--game_server_port',default=18001)
    parser.add_argument('--machine_server_ip',default="")
    parser.add_argument('--machine_server_port',default=0)
    parser.add_argument('--user_name',default='qmx4pp2y9ujb5')
    parser.add_argument('--episode_horizon',default=100000)
    
    # Train Configuration
    parser.add_argument('--batch_size',default=64), 
    parser.add_argument('--ctx_size',default=8), 
    parser.add_argument('--lr',default=5e-4), 
    parser.add_argument('--n_episodes',default=99999999), 
    parser.add_argument('--n_workers',default=8), 
    parser.add_argument('--rollout_steps',default=128), 
    parser.add_argument('--dtype',default=torch.float32) 
    
    arg_config = parser.parse_args()
    config = vars(arg_config)
    
    return config