import argparse

from gen_training_data_zelda import generate_training_data_zelda
from gen_training_data_lr import generate_training_data_lr
from gen_training_data_lego import generate_training_data_lego

from train_zelda import train_zelda
# from .train import train_model
# from .inference import infer

from inference_zelda import inference_zelda


def main(domain, mode, username, debug):
    if domain == "zelda": 
        generate_training_data_zelda(domain, mode, username, debug)
        train_zelda(domain, mode, username, debug)
        inference_zelda(domain, mode, username, debug)
        # TODO: inference_zelda here
    elif domain == "lr": 
        generate_training_data_lr(domain, mode)
        # TODO: train_lr here
        # TODO: inference_lr here
    elif domain == "lego": 
        generate_training_data_lego(domain, mode)
        # TODO: train_lego here
        # TODO: inference_lego here



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", type=str, choices=["zelda", "lr", "lego"])
    parser.add_argument("-m", "--mode", type=str, choices=["controllable", "non_controllable"])
    parser.add_argument("-u", "--username", type=str, default="ms12010")
    parser.add_argument("-l", "--debug", action='store_true')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args.domain, args.mode, args.username, args.debug)