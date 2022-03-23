from .all import AllDataset, collate_fn as all_collate


class Dataset:
    _d = {
        'all': (AllDataset, all_collate),
    }

    @classmethod
    def from_config(cls, config):
        Dataset, collate_fn = cls._d[config.mode]
        return Dataset(config), collate_fn

