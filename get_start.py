import os
from pathlib import Path
import time
import random
import numpy as np
from Utils.config import readParser

from DunkCityDynasty.env.gym_env import GymEnv


class RandomAgent():
    def __init__(self):
        pass

    def get_actions(self, states):
        return {key: np.random.randint(0, 8) for key in states}


def main():
    # env config
    # --- win env
    # config = {
    #     'id': 1,
    #     'env_setting': 'win',
    #     'client_path': 'path-to-game-client',
    #     'rl_server_ip': '127.0.0.1',
    #     'rl_server_port': 42636,
    #     'game_server_ip': '127.0.0.1',
    #     'game_server_port': 18000,
    #     'machine_server_ip': '',
    #     'machine_server_port': 0,
    #     'user_name': 'xxxx',
    #     'episode_horizon': 100000
    # }

    # --- linux env
    # config = {
    #     'id': 1,
    #     'env_setting': 'linux',
    #     'client_path': os.path.join(Path(__file__).parent.parent.resolve(), 'game_package_release'),
    #     'rl_server_ip': '127.0.0.1',
    #     'rl_server_port': 42636,
    #     'game_server_ip': '121.40.214.152',
    #     'game_server_port': 18001,
    #     'machine_server_ip': '',
    #     'machine_server_port': 0,
    #     'user_name': 'qmx4pp2y9ujb5',
    #     'episode_horizon': 100000
    # }

    # # --- multi_machine
    # config = {
    #     'id': 1,
    #     'env_setting': 'multi_machine',
    #     'client_path': '',
    #     'rl_server_ip': '10.219.204.81',
    #     'rl_server_port': 42636,
    #     'game_server_ip': '127.0.0.1',
    #     'game_server_port': 18000,
    #     'machine_server_ip': '10.219.204.76',
    #     'machine_server_port': 6667,
    #     'user_name': 'xxxx',
    #     'episode_horizon': 100000
    # }

    config = readParser()
    config['id'] = 0
    config['rl_server_port'] = config['rl_server_ports'][config['id']]
    env = GymEnv(config)
    agent = RandomAgent()
    user_name = "qmx4pp2y9ujb5"
    states, infos = env.reset(user_name = user_name, render = True)
    while True:
        actions = agent.get_actions(states)
        states, rewards, dones, truncated, infos = env.step(actions)
        print(actions)
        if dones['__all__']:
            break

if __name__ == '__main__':
    main()