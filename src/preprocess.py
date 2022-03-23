from argparse import ArgumentParser
from omegaconf import OmegaConf
from modules import PreProcessor


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    preprocessor = PreProcessor.from_config(config)
    preprocessor.run()


if __name__ == '__main__':
    main()
