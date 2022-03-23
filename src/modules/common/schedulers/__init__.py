from .noam import NoamLR


class Scheduler:
    _d = {
        'noam': NoamLR
    }

    @classmethod
    def from_config(cls, optimizer, config, last_epoch):
        return cls._d[config.mode](optimizer, **config, last_epoch=last_epoch)
