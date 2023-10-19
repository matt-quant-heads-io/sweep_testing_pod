import argparse

from gen_training_data import generate_training_data
# from .train import train_model
# from .inference import infer


def main(domain):
    generate_training_data(domain)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", type=str, choices=["zelda", "lr", "lego"])
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args.domain)