import os
from shutil import copyfile
import argparse
from solver import Solver
from torch.backends import cudnn
import json


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if not os.path.exists(os.path.join(config.output_dir, 'logs')):
        os.makedirs(os.path.join(config.output_dir, 'logs'))

    if not os.path.exists(os.path.join(config.output_dir, 'models')):
        os.makedirs(os.path.join(config.output_dir, 'models'))
    if not os.path.exists(os.path.join(config.output_dir, 'samples')):
        os.makedirs(os.path.join(config.output_dir, 'samples'))
    if not os.path.exists(os.path.join(config.output_dir, 'results')):
        os.makedirs(os.path.join(config.output_dir, 'results'))
    if not os.path.exists(os.path.join(config.output_dir, 'source-code')):
        os.makedirs(os.path.join(config.output_dir, 'source-code'))

    with open(os.path.join(config.output_dir, 'source-code/hyperparams.json'), 'w') as fp:
        json.dump(vars(config), fp)

    copyfile('main.py', os.path.join(config.output_dir, 'source-code', 'main.py'))
    copyfile('solver.py', os.path.join(config.output_dir, 'source-code', 'solver.py'))

    solver = Solver(config)

    if config.mode == 'train':
        solver.train()
    if 'edit' in config.mode:
        solver.restore_model(solver.resume_iters)

        from config.config_attributes import directions
        directions = directions[config.model_name]

        if not os.path.exists(f'./fake/original'):
            os.makedirs(f'./fake/original')
        if not os.path.exists(f'./fake/{config.attribute_to_edit}'):
            os.makedirs(f'./fake/{config.attribute_to_edit}')

        if config.mode == 'edit_modewise':
            direction = directions['linear_attributes'][config.attribute_to_edit]
            print(f'Generating linear edits {config.attribute_to_edit}...')

            solver.generate_edit(direction, config.attribute_to_edit, linear=True, n=20)
        if config.mode == 'edit_multilinear':
            direction = directions['multilinear_attributes'][config.attribute_to_edit]
            print(f'Generating multilinear edits {config.attribute_to_edit}...')

            solver.generate_edit(direction, config.attribute_to_edit, linear=False, n=20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--image_size', type=int, default=512, help='image resolution')

    parser.add_argument('--num_attributes', type=int, default=3, help='num attributes')
    parser.add_argument('--num_classes', type=str, default="5,2,4", help='num attributes')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--n_batches', type=int, default=1, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000001, help='number of total iterations for training D')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for regressor')
    parser.add_argument('--resume_iters', type=str, default=None, help='resume training from this step')
    parser.add_argument('--use_multiple_gpus', type=str2bool, default=False, help='use multiple gpus')

    parser.add_argument('--penalty_lam', type=float, default=0.0001, help='use multiple gpus')

    parser.add_argument('--ranks', type=str, default="512,4,4", help='num attributes')
    parser.add_argument('--cp_rank', type=int, default=0, help='num attributes')
    parser.add_argument('--tucker_ranks', type=str, default="0,0,0,0", help='num attributes')
    parser.add_argument('--components', type=str, default="512,4,4", help='num attributes')
    parser.add_argument('--model_name', type=str, default="pggan_celebahq1024", help='')

    parser.add_argument('--test', type=str2bool, default=False, help='use multiple gpus')
    parser.add_argument('--edit_directly', type=str2bool, default=False, help='(for stylegan) edit the tensor directly?')

    # Editing configuration.
    parser.add_argument('--attribute_to_edit', type=str, default="blond", help='num attributes')
    parser.add_argument('--n_to_edit', type=int, default=20, help='number of samples to generate edited by the target attribute')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'edit_modewise', 'edit_multilinear'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--image_dir', type=str, default='./ibug-combined')
    parser.add_argument('--output_dir', type=str, default='./samples')
    parser.add_argument('--model_dir', type=str, default='./weights-new')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=250)
    parser.add_argument('--model_save_step', type=int, default=50000)

    config = parser.parse_args()
    print(config)
    main(config)
