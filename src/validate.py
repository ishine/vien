from argparse import ArgumentParser
from omegaconf import OmegaConf

from modules.validate import validate


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    validate(args, config)


if __name__ == '__main__':
    main()
