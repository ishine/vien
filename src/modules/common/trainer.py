from abc import abstractmethod
from torch.utils.tensorboard import SummaryWriter


class TrainerBase:
    @abstractmethod
    def run(self):
        pass

    def log(self, epoch, loggers, tracker, mode='train'):
        for logger in loggers:
            if isinstance(logger, dict):
                logger[mode].log({k: v.mean() for k, v in tracker.items()})
            elif isinstance(logger, SummaryWriter):
                for k, v in tracker.items():
                    logger.add_scalar(f'{mode}/{k}', v.mean(), epoch)
