import time
import torch.nn as nn
import numpy as np
import tensorly as tl
import torch
import os

from tensorly.tenalg import inner
from utils import load_generator

from utils import postprocess
from imageio import imsave

tl.set_backend('pytorch')


class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Model(nn.Module):
    def __init__(self, s1, s2, s3, r1, r2, r3, tucker_ranks, cp_rank, inv_bases, bases, device):
        super(Model, self).__init__()
        self.cp_rank = cp_rank

        self.register_buffer('U_T1', tl.tensor(inv_bases[0]).to(device))
        self.register_buffer('U_T2', tl.tensor(inv_bases[1]).to(device))
        self.register_buffer('U_T3', tl.tensor(inv_bases[2]).to(device))

        self.register_buffer('U1', tl.tensor(bases[0]).to(device))
        self.register_buffer('U2', tl.tensor(bases[1]).to(device))
        self.register_buffer('U3', tl.tensor(bases[2]).to(device))

        # Add and register the factors
        self.factors = nn.ParameterList()

        print('--------')
        if self.cp_rank > 0:
            # CP
            print('Using CP')
            for index, (in_size, rank) in enumerate(zip([r1, r2, r3, 512], [cp_rank] * 4)):
                init = torch.ones([in_size, rank])
                init.data.uniform_(-0.1, 0.1)

                self.factors.append(nn.Parameter(init, requires_grad=True))
        else:
            # TUCKER
            print('Using tucker')
            core = torch.ones(tucker_ranks)
            core.data.uniform_(-0.1, 0.1)
            self.core = nn.Parameter(core, requires_grad=True)

            for index, (in_size, rank) in enumerate(zip([s1, s2, s3, 512], tucker_ranks)):
                init = torch.ones([in_size, rank])
                init.data.uniform_(-0.1, 0.1)

                self.factors.append(nn.Parameter(init, requires_grad=True))

        print('--------')

    def forward(self, z):
        # form the regression tensor
        regression_weights = tl.cp_to_tensor((None, self.factors)) if self.cp_rank > 0 else tl.tucker_to_tensor((self.core, self.factors))

        # generalized inner product
        out = inner(z, regression_weights, n_modes=tl.ndim(z) - 1)

        return out

    def penalty(self, order=2, cp=True):
        # l2 reg
        penalty = 0 if cp else torch.sum(self.core ** 2)  # core tensor penalty
        penalty += torch.sum(torch.stack([torch.sum(f ** 2) for f in self.factors]))

        return penalty


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, config):
        """Initialize configurations."""
        # Model configurations.
        self.config = config
        self.image_size = config.image_size
        self.num_attributes = config.num_attributes
        self.num_classes = [int(x) for x in config.num_classes.split(',')]

        # Training configurations.
        self.batch_size = config.batch_size
        self.n_batches = config.n_batches
        self.num_iters = config.num_iters
        self.lr = config.lr
        self.resume_iters = config.resume_iters
        self.use_multiple_gpus = config.use_multiple_gpus
        self.edit_directly = config.edit_directly
        self.model_name = config.model_name
        self.cp_rank = config.cp_rank
        self.tucker_ranks = [int(x) for x in config.tucker_ranks.split(',')]

        self.penalty_lam = config.penalty_lam
        self.ranks = [int(x) for x in config.ranks.split(',')]
        self.components = [int(x) for x in config.components.split(',')]
        self.test = config.test

        self.path = '{}_cprank-{}_tuckerrank-{}_pcaranks-{}_penalty-{}'.format(self.model_name, self.cp_rank, self.tucker_ranks, self.ranks, self.penalty_lam)

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print('-------------------')
        print('GPU?: {}'.format(torch.cuda.is_available()))
        print('-------------------')
        if self.device.type == 'cuda':
            print('-------------------')
            print(torch.cuda.get_device_name(0))
            print('-------------------')

        # Directories.
        self.log_dir = os.path.join(config.output_dir, 'logs')
        self.sample_dir = os.path.join(config.output_dir, 'samples')
        self.model_save_dir = config.model_dir
        self.results_dir = os.path.join(config.output_dir, 'results')
        self.output_dir = config.output_dir
        self.image_dir = config.image_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Build the model and the regressor"""

        self.s1 = self.components[0]
        self.s2 = self.components[1]
        self.s3 = self.components[2]

        self.r1 = self.ranks[0]
        self.r2 = self.ranks[1]
        self.r3 = self.ranks[2]

        self.generator = load_generator(self.model_name)

        if torch.cuda.device_count() > 1 and self.use_multiple_gpus:
            print('----------------------------')
            print('Using {} GPUs'.format(torch.cuda.device_count()))
            print('----------------------------')

            self.generator = MyDataParallel(self.generator)

        self.generator.to(self.device)
        self.gan_type = self.model_name.split('_')[0]

        bases_init = [torch.zeros(rank, rank) for rank in [self.r1, self.r2, self.r3]]
        self.Model = Model(self.s1, self.s2, self.s3, self.r1, self.r2, self.r3, tucker_ranks=self.tucker_ranks, cp_rank=self.cp_rank, inv_bases=bases_init, bases=bases_init, device=self.device)

        if torch.cuda.device_count() > 1 and self.use_multiple_gpus:
            print('----------------------------')
            print('Using {} GPUs'.format(torch.cuda.device_count()))
            print('----------------------------')

            self.Model = MyDataParallel(self.Model)

        self.Model.to(self.device)
        self.print_network(self.Model, 'Model')

        print('--------------------------------------------------------')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator, discriminator, and classifiers."""
        path = '{}/{}-{}.ckpt'.format('./weights', self.path, resume_iters)
        print('Loading the trained models from step from {}...'.format(path))

        check = torch.load('{}/{}-{}.ckpt'.format(self.model_save_dir, self.path, resume_iters), map_location=lambda storage, loc: storage)
        self.Model.load_state_dict(check, strict=True)

    def generate_edit(self, direction, direction_name, linear, n=5):
        np.random.seed(0)
        torch.manual_seed(0)
        noise = torch.randn(n, self.generator.z_space_dim)

        for i, n in enumerate(noise):
            #################
            # 1) Build the edit tensors
            #################

            # define empty selector tensors
            editU_T1 = torch.zeros((self.ranks[0], self.ranks[1], self.ranks[2])).to(self.device)
            editU_T2 = torch.zeros((self.ranks[0], self.ranks[1], self.ranks[2])).to(self.device)
            editU_T3 = torch.zeros((self.ranks[0], self.ranks[1], self.ranks[2])).to(self.device)
            editU_T123 = torch.zeros((self.ranks[0], self.ranks[1], self.ranks[2])).to(self.device)

            if linear:
                # channels
                edit_unf = tl.unfold(editU_T1, 0)
                edit_unf[direction[0][0], :] = direction[0][1]
                editU_T1 = tl.fold(edit_unf, 0, editU_T1.shape)

                # 1st spatial dimension
                edit_unf = tl.unfold(editU_T2, 1)
                edit_unf[direction[1][0], :] = direction[1][1]
                editU_T2 = tl.fold(edit_unf, 1, editU_T2.shape)

                # 2nd spatial dimension
                edit_unf = tl.unfold(editU_T3, 2)
                edit_unf[direction[2][0], :] = direction[2][1]
                editU_T3 = tl.fold(edit_unf, 2, editU_T3.shape)
            else:
                # multilinear term
                edit_unf = tl.tensor_to_vec(editU_T123)
                edit_unf[direction[0]] = direction[1]
                editU_T123 = tl.vec_to_tensor(edit_unf, editU_T123.shape)

            #################
            # 2) Regress back to latent space and edit
            #################

            # build the Z' edit tensor
            Z_prime = (
                tl.tenalg.mode_dot(editU_T1, self.Model.U1, 0) +
                tl.tenalg.mode_dot(editU_T2, self.Model.U2, 1) +
                tl.tenalg.mode_dot(editU_T3, self.Model.U3, 2) +
                tl.tenalg.multi_mode_dot(editU_T123, [self.Model.U1, self.Model.U2, self.Model.U3], modes=[0, 1, 2])
            )
            # regress back to original latent space
            z_prime = self.Model(Z_prime[None, :])

            if self.gan_type == 'pggan':
                x = postprocess(self.generator(n[None, :], start=2)['image'])[0]
                x_prime = postprocess(self.generator(n[None, :] + z_prime, start=2)['image'])[0]

            elif self.gan_type == 'stylegan':
                n = self.generator.mapping(n[None, :])['w']
                n_edit = n + z_prime

                n_edit = self.generator.truncation(n_edit, trunc_psi=0.7, trunc_layers=8)
                n_orig = self.generator.truncation(n, trunc_psi=0.7, trunc_layers=8)

                # partial forward pass, get intermediate activations
                z = self.generator.synthesis(n_edit, start=2, stop=3)['x']

                x = postprocess(self.generator.synthesis(n_orig, start=2)['image'])[0]
                x_prime = postprocess(self.generator.synthesis(n_edit, x=z + Z_prime, start=2 if linear else 3)['image'])[0]

            if not os.path.exists(f'./fake_{self.model_name}/{direction_name}/'):
                os.makedirs(f'./fake_{self.model_name}/{direction_name}/')

            if not os.path.exists(f'./fake_{self.model_name}/original/'):
                os.makedirs(f'./fake_{self.model_name}/original/')

            edit_path = f'./fake_{self.model_name}/{direction_name}/{str(i).zfill(4)}.jpg'
            real_path = f'./fake_{self.model_name}/original/{str(i).zfill(4)}.jpg'
            imsave(real_path, x)
            imsave(edit_path, x_prime)

            print(f'edited image saved to {edit_path}')

    def get_mpca_transforms(self, n_components=[512, 4, 4], ranks=[512, 4, 4]):
        """
        Each mode-n unfolding's eigenvectors are computed
        """
        inv_bases = [torch.eye(n, device=self.device)[:, :k].T for n, k in zip(n_components, ranks)]
        bases = [torch.eye(n, device=self.device)[:, :k] for n, k in zip(n_components, ranks)]

        loop = True

        # one-shot process if we don't reduce dimensions. Otherwise, perform APP scheme
        for n in range(1 if self.ranks[0] == self.components[0] else 10):
            for mode in range(len(self.z.shape[1:])):
                # INIT: full projection
                X_partial = self.z if n == 0 else tl.tenalg.multi_mode_dot(self.z, inv_bases, modes=[1, 2, 3], skip=mode)
                print(f'computing partial {n}, mode {mode}/3')

                # note: less efficient to loop over the samples this way, but means we can have a larger batch
                if loop:
                    scat = 0
                    for _, x in enumerate(X_partial):
                        m_unfold = tl.unfold(x, mode)
                        scat += m_unfold @ m_unfold.T
                    scat /= self.z.shape[0]
                else:
                    m_unfold = tl.unfold(self.z, mode + 1)
                    scat = (m_unfold @ m_unfold.T) / self.z.shape[0]

                # covariance matrix is positive semi-def, so eigdecomp is same as SVD
                U, S, V = torch.svd(scat)
                U = U[:, :ranks[mode]]

                inv_bases[mode] = U.T
                bases[mode] = U

        return inv_bases, bases

    def train(self):
        # initialise the M set of activations to zeros
        self.z = torch.zeros((self.batch_size * self.n_batches, self.components[0], self.components[1], self.components[2]), device=self.device)

        with torch.no_grad():
            for i in range(self.n_batches):
                print(f'doin  batch {i}')
                np.random.seed(i)
                torch.manual_seed(i)

                if self.gan_type == 'pggan':
                    self.noise = torch.randn(self.batch_size, self.generator.z_space_dim)
                    self.noise = self.noise.to(self.device)
                    self.noise = self.generator.layer0.pixel_norm(self.noise)

                    self.z[(self.batch_size * i):(self.batch_size * (i + 1))] = self.generator(self.noise, start=2, stop=3)['x']

                elif self.gan_type in ['stylegan']:
                    self.noise = torch.randn(self.batch_size, self.generator.z_space_dim)
                    self.noise = self.noise.to(self.device)

                    self.noise = self.generator.mapping(self.noise)['w']
                    self.noise = self.generator.truncation(self.noise, trunc_psi=1.0, trunc_layers=18)

                    self.z[(self.batch_size * i):(self.batch_size * (i + 1))] = self.generator.synthesis(self.noise, start=2, stop=3)['x']

            # zero mean
            self.z -= torch.mean(self.z, axis=0).detach()

            # compute the bases
            self.inv_bases, self.bases = self.get_mpca_transforms(ranks=[self.r1, self.r2, self.r3], n_components=self.z.shape[1:])

        self.w_optimizer = torch.optim.Adam(self.Model.parameters(), self.lr)

        if self.resume_iters:
            self.restore_model(self.resume_iters)

        self.recon_batch_size = 16
        criterion = torch.nn.MSELoss()

        # Learn the regression
        for i in range(0, self.num_iters):
            t = 1000 * time.time()
            np.random.seed(int(t) % 2**32)
            torch.manual_seed(int(t) % 2**32)

            if torch.cuda.is_available() or not self.resume_iters:

                #####################
                # BEGIN (forward pass)
                if self.gan_type == 'pggan':
                    noise = torch.randn(self.recon_batch_size, self.generator.z_space_dim)
                    noise = noise.to(self.device).detach()

                    noise_norm = self.generator.layer0.pixel_norm(noise)
                    z = self.generator(noise_norm, start=2, stop=3)['x'].detach()

                elif self.gan_type in ['stylegan']:
                    noise = torch.randn(self.recon_batch_size, self.generator.z_space_dim)
                    noise = noise.to(self.device)

                    noise = self.generator.mapping(noise)['w']

                    noise_trunc = self.generator.truncation(noise, trunc_psi=1.0, trunc_layers=18)
                    z = self.generator.synthesis(noise_trunc, start=2, stop=3)['x']
                # END (forward pass)
                #####################

                # regress tensor Z -> z to latent code
                out = self.Model(z)

                # and penalise the reconstruction
                recon_loss = criterion(out, noise)

                penalty_loss = self.penalty_lam * self.Model.penalty(2)

                if i % 1000 == 0 or not torch.cuda.is_available():
                    print(i, 'LOSSES', 'recon:', recon_loss.item(), 'penalty:', penalty_loss.item())

                w_loss = recon_loss + penalty_loss

                self.w_optimizer.zero_grad()
                w_loss.backward()
                self.w_optimizer.step()

            # =================================================================================== #
            #                                 5. Miscellaneous                                    #
            # =================================================================================== #
            # save checkpoints
            if i % 1000 == 0 and i > 0:
                Model_path = os.path.join(self.model_save_dir, '{}-{}.ckpt'.format(self.path, i))
                torch.save(self.Model.state_dict(), Model_path)
