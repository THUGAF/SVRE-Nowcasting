import os
import argparse
import warnings
import torch

from model import *
from utils.trainer import Assimilation
import utils.dataloader as dataloader


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

# input and output settings
parser.add_argument('--data-path', type=str, default='/data/gaf/SBandCRUnzip')
parser.add_argument('--output-path', type=str, default='results/Assimilation')
parser.add_argument('--input-steps', type=int, default=10)
parser.add_argument('--forecast-steps', type=int, default=10)
parser.add_argument('--sample-index', type=int, default=16840)
parser.add_argument('--seed', type=int, default=2022)

# nowcasting settings
parser.add_argument('--resolution', type=float, default=6.0, help='Time resolution (min)')
parser.add_argument('--lon-range', type=int, nargs='+', default=[271, 527])
parser.add_argument('--lat-range', type=int, nargs='+', default=[335, 591])
parser.add_argument('--vmax', type=float, default=70.0)
parser.add_argument('--vmin', type=float, default=0.0)

# evaluation settings
parser.add_argument('--thresholds', type=int, nargs='+', default=[10, 15, 20, 25, 30, 35, 40])

args = parser.parse_args()


def main():
    # Display global settings
    print('Temporal resolution: {} min'.format(args.resolution))
    print('Spatial resolution: 1.0 km')
    print('Input steps: {}'.format(str(args.input_steps)))
    print('Forecast steps: {}'.format(str(args.forecast_steps)))
    print('Input time range: {} min'.format(str(args.input_steps * args.resolution)))
    print('Forecast time range: {} min'.format(str(args.forecast_steps * args.resolution)))

    # fix the random seed
    torch.manual_seed(args.seed)
    
    # Set the model
    model = AttnUNet(args.input_steps, args.forecast_steps)

    sample_loader = dataloader.load_sample(args.data_path, args.sample_index, args.input_steps,
                                           args.forecast_steps, args.lon_range, args.lat_range)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    trainer = Assimilation(args)
    trainer.predict(model, sample_loader)


if __name__ == '__main__':
    main()
