from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from .model import TTSModel
from .gan import MultiPeriodDiscriminator
from .loss import discriminator_loss, generator_loss, fm_loss
from .utils import slice_segments
from ..common.trainer import TrainerBase
from ..common.loggers.csv import CSVLogger
from ..common.utils import Tracker, seed_everything
from ..common.datasets import Dataset
from ..common.schedulers import NoamLR
from ..common.transforms import MelSpectrogram


class Trainer(TrainerBase):
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        OmegaConf.save(self.config, self.output_dir / 'ttslearn.yaml')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator = TTSModel(config.model).to(self.device)
        self.discriminator = MultiPeriodDiscriminator().to(self.device)

        self.to_mel = MelSpectrogram(**config.mel).to(self.device)

    def run(self):
        seed_everything(self.config.seed)

        csv_logger = CSVLogger(self.output_dir / 'logs/train.csv')
        tb_logger = SummaryWriter(log_dir=str(self.output_dir / 'logs'))

        train_dl, valid_dl = self.setup_data()

        g = self.generator
        d = self.discriminator

        opt_g = torch.optim.AdamW(g.parameters(), eps=1e-9, **self.config.optimizer)
        opt_d = torch.optim.AdamW(d.parameters(), eps=1e-9, **self.config.optimizer)

        start_epoch = 0
        if (self.output_dir / 'last.ckpt').exists():
            d = torch.load(self.output_dir / 'last.ckpt')
            start_epoch = d['epoch'] + 1
            g.load_state_dict(d['g'])
            d.load_state_dict(d['d'])
            opt_g.load_state_dict(d['opt_g'])
            opt_d.load_state_dict(d['opt_d'])

        sche_g = NoamLR(opt_g, **self.config.scheduler, last_epoch=start_epoch-1)
        sche_d = NoamLR(opt_d, **self.config.scheduler, last_epoch=start_epoch-1)

        for e in range(start_epoch, self.config.train.num_epochs):
            g.train()
            d.train()
            tracker = Tracker()
            bar = tqdm(desc=f'Train Epoch: {e}', total=len(train_dl))
            for batch in train_dl:
                opt_d.zero_grad()
                loss_dict = self.handle_batch_gan(g, d, opt_g, opt_d, sche_g, sche_d, batch, train=True)
                tracker.update(**loss_dict)
                bar.update()
                self.set_losses(bar, tracker)
            self.log(e, [csv_logger, tb_logger], tracker, mode='train')
            bar.close()

            g.eval()
            d.eval()
            tracker = Tracker()
            bar = tqdm(desc=f'Valid Epoch: {e}', total=len(valid_dl))
            for batch in valid_dl:
                with torch.no_grad():
                    loss_dict = self.handle_batch_gan(g, d, opt_g, opt_d, sche_g, sche_d, batch, train=False)
                tracker.update(**loss_dict)
                bar.update()
                self.set_losses(bar, tracker)
            self.log(e, [csv_logger, tb_logger], tracker, mode='valid')
            bar.close()

            if (e + 1) % self.config.train.save_interval == 0:
                self.save(self.config, e, g, d, opt_g, opt_d, f'{e+1}.ckpt')
            self.save(self.config, e, g, d, opt_g, opt_d, f'last.ckpt')

    def handle_batch_gan(self, g, d, opt_g, opt_d, sche_g, sche_d, batch, train=True):
        batch = [b.to(self.device) for b in batch]
        (
            _,
            _,
            wav,
            _,
            mel,
            _,
            _,
            _,
            _
        ) = batch
        y_hat, ids_slice, loss_dict = g.compute_loss(batch)
        y_mel = slice_segments(mel, ids_slice, segment_size=self.config.model.mel_segment)
        y_hat_mel = self.to_mel(y_hat.squeeze(1))
        y = slice_segments(wav, ids_slice, segment_size=self.config.model.segment_size)

        real, fake, _, _ = d(y, y_hat.detach())
        loss_d = discriminator_loss(real, fake)
        if train:
            opt_d.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(d.parameters(), max_norm=5)
            opt_d.step()
            sche_d.step()

        _, fake, f_real, f_fake = d(y, y_hat)
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45
        loss_fm = fm_loss(f_real, f_fake)
        loss_gen = generator_loss(fake)
        loss_g = loss_gen + loss_fm + loss_mel + loss_dict['loss']
        if train:
            opt_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), max_norm=5)
            opt_g.step()
            sche_g.step()
        return dict(
            d=loss_d,
            g_gan=loss_gen,
            mel=loss_mel,
            duration=loss_dict['duration'],
            pitch=loss_dict['pitch']
        )

    def save(self, config, epoch, g, d, opt_g, opt_d, fn):
        torch.save({
            'config': config,
            'epoch': epoch,
            'g': g.state_dict(),
            'd': d.state_dict(),
            'opt_g': opt_g.state_dict(),
            'opt_d': opt_d.state_dict()
        }, self.output_dir / fn)

    def set_losses(self, bar, tracker):
        bar.set_postfix_str(f', '.join([f'{k}: {v.mean():.6f}' for k, v in tracker.items()]))

    def setup_data(self):
        ds, collate_fn = Dataset.from_config(self.config.data)
        train_ds = Subset(ds, list(range(self.config.data.valid_size, len(ds))))
        train_dl = DataLoader(
            train_ds,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_fn
        )
        valid_ds = Subset(ds, list(range(self.config.data.valid_size)))
        valid_dl = DataLoader(
            valid_ds,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn
        )
        return train_dl, valid_dl
