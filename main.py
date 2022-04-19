import os
import argparse
import warnings

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models import Generator, GAN
import utils.dataloader as dataloader


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='DL-Nowcasting')

# global settings
parser.add_argument('--data-path', type=str, default='/data/gaf/SBandCRNpz')
parser.add_argument('--output-path', type=str, default='results/AttnUNet_GAN_CR')

# data loading settings
parser.add_argument('--train-ratio', type=float, default=0.7)
parser.add_argument('--valid-ratio', type=float, default=0.1)
parser.add_argument('--start-point', type=int, default=0)
parser.add_argument('--end-point', type=int, default=10000)
parser.add_argument('--sample-point', type=int, default=0)

# training settings
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--early-stopping', action='store_true')
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--max-iterations', type=int, default=100000)
parser.add_argument('--start-iterations', type=int, default=0)
parser.add_argument('--gen-lr', type=float, default=1e-4)
parser.add_argument('--disc-lr', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--recon-reg', type=float, default=10)
parser.add_argument('--global-var-reg', type=float, default=0.1)
parser.add_argument('--local-var-reg', type=float, default=0)
parser.add_argument('--num-sampling', type=int, default=1)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--log-interval', type=int, default=1)
parser.add_argument('--seed', type=int, default=2021)

# input and output settings
parser.add_argument('--input-steps', type=int, default=10)
parser.add_argument('--forecast-steps', type=int, default=10)

# model settings
parser.add_argument('--generator', type=str, choices=['EncoderForecaster', 'SmaAt_UNet', 'AttnUNet'], 
                    default='AttnUNet')
parser.add_argument('--add-gan', action='store_true')
parser.add_argument('--lon-range', type=int, nargs='+', default=[273, 529])
parser.add_argument('--lat-range', type=int, nargs='+', default=[270, 526])
parser.add_argument('--in-channels', type=int, default=1)
parser.add_argument('--out-channels', type=int, default=1)
parser.add_argument('--vmax', type=float, default=80.0)
parser.add_argument('--vmin', type=float, default=0.0)

# Encoder-Forecaster settings
parser.add_argument('--hidden-channels', type=int, nargs='+', default=[32, 64, 128])

# evaluation settings
parser.add_argument('--thresholds', type=int, nargs='+', default=[10, 15, 20, 25, 30, 35, 40])

args = parser.parse_args()


def main():
    # fix the random seed
    pl.seed_everything(args.seed)

    if args.train + args.test + args.predict == 0:
        raise NotImplementedError('''No mode has been choosed! Turn the modes on by setting 
            `args.train`, `args.test`, or `args.predict` manually.''')

    # set model and optimizer
    if args.add_gan:
        model = GAN(args)
    else:
        model = Generator(args)

    # train, validate and test
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_path,
        monitor='val/g_loss',
        filename=args.generator if isinstance(model, Generator) else args.generator + '_GAN',
        save_top_k=1,
        every_n_train_steps=args.log_interval
    )
    callbacks = [checkpoint_callback]

    if args.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor='val/g_loss',
            patience=10
        )
        callbacks.append(early_stopping_callback)

    logger = TensorBoardLogger(
        save_dir='logs',
        name='',
        version=1 if args.add_gan else 0,
        default_hp_metric=False
    )

    if args.pretrain:
        resume_from_checkpoint = os.path.join(checkpoint_callback.dirpath, checkpoint_callback.filename) + '.ckpt'
    else:
        resume_from_checkpoint = None

    trainer = Trainer(
        gpus=1,
        max_steps=args.max_iterations,
        log_every_n_steps=args.log_interval,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=0,
        callbacks=callbacks,
        logger=logger,
        resume_from_checkpoint=resume_from_checkpoint
    )

    # load data
    if args.train or args.test:
        train_loader, val_loader, test_loader = dataloader.load_data(
            args.data_path, args.start_point, args.end_point, args.batch_size,
            args.num_workers, args.train_ratio, args.valid_ratio,
            args.lon_range, args.lat_range, args.seed
        )

    # train
    if args.train:
        trainer.fit(model, train_loader, val_loader)

    # test
    if isinstance(model, Generator):
        ckpt_path = os.path.join(
            args.output_path, args.generator + '.ckpt')
    else:
        ckpt_path = os.path.join(args.output_path, args.generator + '_GAN.ckpt')
    
    if args.test:
        trainer.test(model, test_loader, ckpt_path=ckpt_path)

    # predict
    if args.predict:
        sample_loader = dataloader.load_sample(
            args.data_path, args.sample_point, args.lon_range, args.lat_range, args.seed
        )
        trainer.predict(model, sample_loader, ckpt_path=ckpt_path)

    print('\nAll tasks done.')


if __name__ == "__main__":
    main()
