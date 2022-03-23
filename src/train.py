from argparse import ArgumentParser
from omegaconf import OmegaConf

from modules import Trainer


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    trainer = Trainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
