import sys
import os
import os.path as osp
import clip
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import torch.backends.cudnn as cudnn
import numpy as np
import time
import shutil
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from .aux import TrainingStatTracker, update_progress, update_stdout, sec2dhms
from .config import SEMANTIC_DIPOLES_CORPORA, GENFORCE_MODELS


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Trainer(object):
    def __init__(self, params=None, exp_dir=None, use_cuda=False, multi_gpu=False):
        if params is None:
            raise ValueError("Cannot build a Trainer instance with empty params: params={}".format(params))
        else:
            self.params = params
        self.use_cuda = use_cuda
        self.multi_gpu = multi_gpu

        # Use TensorBoard
        self.tensorboard = self.params.tensorboard

        # Set output directory for current experiment (wip)
        self.wip_dir = osp.join("experiments", "wip", exp_dir)

        # Set directory for completed experiment
        self.complete_dir = osp.join("experiments", "complete", exp_dir)

        # Create log sub-directory and define stat.json file
        self.stats_json = osp.join(self.wip_dir, 'stats.json')
        if not osp.isfile(self.stats_json):
            with open(self.stats_json, 'w') as out:
                json.dump({}, out)

        # Create models sub-directory
        self.models_dir = osp.join(self.wip_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        # Define checkpoint model file
        self.checkpoint = osp.join(self.models_dir, 'checkpoint.pt')

        # Setup TensorBoard
        if self.tensorboard:
            # Create tensorboard sub-directory
            self.tb_dir = osp.join(self.wip_dir, 'tensorboard')
            os.makedirs(self.tb_dir, exist_ok=True)
            self.tb = program.TensorBoard()
            self.tb.configure(argv=[None, '--logdir', self.tb_dir])
            self.tb_url = self.tb.launch()
            print("#. Start TensorBoard at {}".format(self.tb_url))
            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

        # Array of iteration times
        self.iter_times = np.array([])

        # Set up training statistics tracker
        self.stat_tracker = TrainingStatTracker()

        # Define cosine similarity loss
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss()

        # Define cross entropy loss
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # Define transform of CLIP image encoder
        self.clip_img_transform = transforms.Compose([transforms.Resize(224),
                                                      transforms.CenterCrop(224),
                                                      transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                           (0.26862954, 0.26130258, 0.27577711))])

    def contrastive_loss(self, img_batch, txt_batch):
        n_img, d_img = img_batch.shape
        n_txt, d_txt = txt_batch.shape

        # Normalise image and text batches
        img_batch_l2 = F.normalize(img_batch, p=2, dim=-1)
        txt_batch_l2 = F.normalize(txt_batch, p=2, dim=-1)

        # Calculate inner product similarity matrix
        similarity_matrix = torch.matmul(img_batch_l2, txt_batch_l2.T)
        labels = torch.arange(n_img)

        return self.cross_entropy_loss(similarity_matrix / self.params.temperature, labels)

    def get_starting_iteration(self, latent_support_sets):
        """Check if checkpoint file exists (under `self.models_dir`) and set starting iteration at the checkpoint
        iteration; also load checkpoint weights to `latent_support_sets`. Otherwise, set starting iteration to 1 in
        order to train from scratch.

        Returns:
            starting_iter (int): starting iteration

        """
        starting_iter = 1
        if osp.isfile(self.checkpoint):
            checkpoint_dict = torch.load(self.checkpoint)
            starting_iter = checkpoint_dict['iter']
            latent_support_sets.load_state_dict(checkpoint_dict['latent_support_sets'])

        return starting_iter

    def log_progress(self, iteration, mean_iter_time, elapsed_time, eta):
        """Log progress (loss + ETA).

        Args:
            iteration (int)        : current iteration
            mean_iter_time (float) : mean iteration time
            elapsed_time (float)   : elapsed time until current iteration
            eta (float)            : estimated time of experiment completion
        """
        # Get current training stats (for the previous `self.params.log_freq` steps) and flush them
        stats = self.stat_tracker.get_means()

        # Update training statistics json file
        with open(self.stats_json) as f:
            stats_dict = json.load(f)
        stats_dict.update({iteration: stats})
        with open(self.stats_json, 'w') as out:
            json.dump(stats_dict, out)

        # Flush training statistics tracker
        self.stat_tracker.flush()

        update_progress("  \\__.Training [bs: {}] [iter: {:06d}/{:06d}] ".format(
            self.params.batch_size, iteration, self.params.max_iter), self.params.max_iter, iteration + 1)
        if iteration < self.params.max_iter - 1:
            print()
        print("         ===================================================================")
        print("      \\__Loss           : {:.08f}".format(stats['loss']))
        print("         ===================================================================")
        print("      \\__Mean iter time : {:.3f} sec".format(mean_iter_time))
        print("      \\__Elapsed time   : {}".format(sec2dhms(elapsed_time)))
        print("      \\__ETA            : {}".format(sec2dhms(eta)))
        print("         ===================================================================")
        update_stdout(8)

    def train(self, generator, latent_support_sets, corpus_support_sets, clip_model):
        """GANxPlainer training function.

        Args:
            generator           : pre-trained GAN generator
            latent_support_sets : TODO: +++
            corpus_support_sets : TODO: +++
            clip_model          : pre-trained CLIP model

        """
        # Save initial `latent_support_sets` model as `latent_support_sets_init.pt`
        torch.save(latent_support_sets.state_dict(), osp.join(self.models_dir, 'latent_support_sets_init.pt'))

        # Save initial `corpus_support_sets` model as `corpus_support_sets_init.pt`
        torch.save(corpus_support_sets.state_dict(), osp.join(self.models_dir, 'corpus_support_sets_init.pt'))

        # Save prompt corpus list to json
        with open(osp.join(self.models_dir, 'semantic_dipoles.json'), 'w') as json_f:
            json.dump(SEMANTIC_DIPOLES_CORPORA[self.params.corpus], json_f)

        # Upload models to GPU if `self.use_cuda` is set (i.e., if args.cuda and torch.cuda.is_available is True).
        if self.use_cuda:
            generator.cuda().eval()
            clip_model.cuda().eval()
            corpus_support_sets.cuda()
            latent_support_sets.cuda().train()
        else:
            generator.eval()
            clip_model.eval()
            latent_support_sets.train()

        # Get latent space dimensionality
        G_dim_z = GENFORCE_MODELS[self.params.gan][1]

        # Set latent support sets (LSS) optimizer
        latent_support_sets_optim = torch.optim.Adam(latent_support_sets.parameters(), lr=self.params.lr)

        # Set learning rate scheduler -- reduce lr after 90% of the total number of training iterations
        latent_support_sets_lr_scheduler = StepLR(optimizer=latent_support_sets_optim,
                                                  step_size=int(0.9 * self.params.max_iter),
                                                  gamma=0.1)

        # Get starting iteration
        starting_iter = self.get_starting_iteration(latent_support_sets)

        # Parallelize models into multiple GPUs, if available and `multi_gpu=True`.
        if self.multi_gpu:
            print("#. Parallelize G and CLIP over {} GPUs...".format(torch.cuda.device_count()))
            # Parallelize generator G
            generator = DataParallelPassthrough(generator)
            # Parallelize CLIP model
            clip_model = DataParallelPassthrough(clip_model)
            # REVIEW: Should I use `cudnn.benchmark`?
            cudnn.benchmark = True

        # Check starting iteration
        if starting_iter == self.params.max_iter:
            print("#. This experiment has already been completed and can be found @ {}".format(self.wip_dir))
            print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
            try:
                shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
                print("  \\__Done!")
            except IOError as e:
                print("  \\__Already exists -- {}".format(e))
            sys.exit()
        print("#. Start training from iteration {}".format(starting_iter))

        # Get experiment's start time
        t0 = time.time()

        # Start training
        for iteration in range(starting_iter, self.params.max_iter + 1):

            # Get current iteration's start time
            iter_t0 = time.time()

            # Set gradients to zero
            generator.zero_grad()
            latent_support_sets.zero_grad()
            clip_model.zero_grad()

            # Sample latent codes from standard Gaussian -- torch.Size([batch_size, G_dim_z])
            z = torch.randn(self.params.batch_size, generator.dim_z)
            if self.use_cuda:
                z = z.cuda()

            # Generate images for the given latent codes
            latent_code = z
            if 'stylegan' in self.params.gan:
                if self.params.stylegan_space == 'W':
                    latent_code = generator.get_w(z, truncation=self.params.truncation)[:, 0, :]
                elif self.params.stylegan_space == 'W+':
                    latent_code = generator.get_w(z, truncation=self.params.truncation)
            img = generator(latent_code)

            # Sample indices of shift vectors (`self.params.batch_size` out of `self.params.num_support_sets`)
            # target_support_sets_indices = torch.randint(0, self.params.num_support_sets, [self.params.batch_size])
            target_support_sets_indices = torch.randint(0, latent_support_sets.num_support_sets, [self.params.batch_size])
            if self.use_cuda:
                target_support_sets_indices = target_support_sets_indices.cuda()

            # Sample shift magnitudes from uniform distributions
            #   U[self.params.min_shift_magnitude, self.params.max_shift_magnitude], and
            #   U[-self.params.max_shift_magnitude, self.params.min_shift_magnitude]
            # Create a pool of shift magnitudes of 2 * `self.params.batch_size` shifts (half negative, half positive)
            # and sample `self.params.batch_size` of them
            shift_magnitudes_pos = (self.params.min_shift_magnitude - self.params.max_shift_magnitude) * \
                torch.rand(target_support_sets_indices.size()) + self.params.max_shift_magnitude
            shift_magnitudes_neg = (self.params.min_shift_magnitude - self.params.max_shift_magnitude) * \
                torch.rand(target_support_sets_indices.size()) - self.params.min_shift_magnitude
            shift_magnitudes_pool = torch.cat((shift_magnitudes_neg, shift_magnitudes_pos))

            shift_magnitudes_ids = torch.arange(len(shift_magnitudes_pool), dtype=torch.float)
            target_shift_magnitudes = shift_magnitudes_pool[torch.multinomial(input=shift_magnitudes_ids,
                                                                              num_samples=self.params.batch_size,
                                                                              replacement=False)]
            if self.use_cuda:
                target_shift_magnitudes = target_shift_magnitudes.cuda()

            # Create support sets mask of size (batch_size, num_support_sets) in the form:
            #       support_sets_mask[i] = [0, ..., 0, 1, 0, ..., 0]
            support_sets_mask = torch.zeros([self.params.batch_size, latent_support_sets.num_support_sets])
            prompt_mask = torch.zeros([self.params.batch_size, 2])
            prompt_sign = torch.zeros([self.params.batch_size, 1])
            if self.use_cuda:
                support_sets_mask = support_sets_mask.cuda()
                prompt_mask = prompt_mask.cuda()
                prompt_sign = prompt_sign.cuda()
            for i, (index, val) in enumerate(zip(target_support_sets_indices, target_shift_magnitudes)):
                support_sets_mask[i][index] += 1.0
                if val >= 0:
                    prompt_mask[i, 0] = 1.0
                    prompt_sign[i] = +1.0
                else:
                    prompt_mask[i, 1] = 1.0
                    prompt_sign[i] = -1.0
            prompt_mask = prompt_mask.unsqueeze(1)

            # Calculate shift vectors for the given latent codes -- in the case of StyleGAN, shifts live in the
            # self.params.stylegan_space, i.e., in Z-, W-, or W+-space. In the Z-/W-space the dimensionality of the
            # latent space is 512. In the case of W+-space, the dimensionality is 512 * (self.params.stylegan_layer + 1)
            if ('stylegan' in self.params.gan) and (self.params.stylegan_space == 'W+'):
                shift = target_shift_magnitudes.reshape(-1, 1) * latent_support_sets(
                    support_sets_mask, latent_code[:, :self.params.stylegan_layer + 1, :].reshape(latent_code.shape[0],
                                                                                                  -1))
            else:
                shift = target_shift_magnitudes.reshape(-1, 1) * latent_support_sets(support_sets_mask, latent_code)

            # Generate images the shifted latent codes
            if ('stylegan' in self.params.gan) and (self.params.stylegan_space == 'W+'):
                latent_code_reshaped = latent_code.reshape(latent_code.shape[0], -1)
                shift = F.pad(input=shift, pad=(0, (17 - self.params.stylegan_layer) * 512), mode='constant', value=0)
                latent_code_shifted = latent_code_reshaped + shift
                latent_code_shifted_reshaped = latent_code_shifted.reshape_as(latent_code)
                img_shifted = generator(latent_code_shifted_reshaped)
            else:
                img_shifted = generator(latent_code + shift)

            # TODO: add comment
            img_pairs = torch.cat([self.clip_img_transform(img), self.clip_img_transform(img_shifted)], dim=0)
            clip_img_pairs_features = clip_model.encode_image(img_pairs)
            clip_img_features, clip_img_shifted_features = torch.split(clip_img_pairs_features, img.shape[0], dim=0)
            clip_img_diff_features = clip_img_shifted_features - clip_img_features

            ############################################################################################################
            ##                                                                                                        ##
            ##                                 Linear Text Paths (StyleCLIP approach)                                 ##
            ##                                                                                                        ##
            ############################################################################################################
            if self.params.styleclip:
                corpus_text_features_batch = torch.matmul(support_sets_mask, corpus_support_sets.SUPPORT_SETS).reshape(
                    -1, 2 * corpus_support_sets.num_support_dipoles, corpus_support_sets.support_vectors_dim)
                corpus_text_features_batch = torch.matmul(prompt_mask, corpus_text_features_batch).squeeze(1)

                # Calculate loss
                if self.params.loss == 'cossim':
                    loss = self.cosine_embedding_loss(clip_img_shifted_features, corpus_text_features_batch,
                                                      torch.ones(corpus_text_features_batch.shape[0]).to(
                                                          'cuda' if self.use_cuda else 'cpu'))
                elif self.params.loss == 'contrastive':
                    loss = self.contrastive_loss(clip_img_shifted_features.float(), corpus_text_features_batch)

            ############################################################################################################
            ##                                                                                                        ##
            ##                                           Linear Text Paths                                            ##
            ##                                                                                                        ##
            ############################################################################################################
            elif self.params.linear:
                corpus_text_features_batch = torch.matmul(support_sets_mask, corpus_support_sets.SUPPORT_SETS).reshape(
                    -1, 2 * corpus_support_sets.num_support_dipoles, corpus_support_sets.support_vectors_dim)

                # Calculate loss
                if self.params.loss == 'cossim':
                    loss = self.cosine_embedding_loss(clip_img_diff_features, prompt_sign * (
                                corpus_text_features_batch[:, 0, :] - corpus_text_features_batch[:, 1,
                                                                      :]) - clip_img_features,
                                                      torch.ones(corpus_text_features_batch.shape[0]).to(
                                                          'cuda' if self.use_cuda else 'cpu'))
                elif self.params.loss == 'contrastive':
                    loss = self.contrastive_loss(clip_img_diff_features.float(), prompt_sign * (
                                corpus_text_features_batch[:, 0, :] - corpus_text_features_batch[:, 1, :]) -
                                                 clip_img_features)

            # TODO: add comment
            ############################################################################################################
            ##                                                                                                        ##
            ##                                                                                                        ##
            ##                                                                                                        ##
            ############################################################################################################
            else:
                # Calculate local text direction using CSS
                local_text_directions = target_shift_magnitudes.reshape(-1, 1) * corpus_support_sets(support_sets_mask,
                                                                                                     clip_img_features)
                # Calculate loss
                if self.params.loss == 'cossim':
                    loss = self.cosine_embedding_loss(clip_img_diff_features, local_text_directions,
                                                      torch.ones(local_text_directions.shape[0]).to(
                                                          'cuda' if self.use_cuda else 'cpu'))
                elif self.params.loss == 'contrastive':
                    loss = self.contrastive_loss(img_batch=clip_img_diff_features.float(),
                                                 txt_batch=local_text_directions)

            # Perform back-prop
            loss.backward()

            # Update weights
            clip_model.float()
            latent_support_sets_optim.step()
            latent_support_sets_lr_scheduler.step()
            clip.model.convert_weights(clip_model)

            # Update statistics tracker
            self.stat_tracker.update(loss=loss.item())

            # Update tensorboard plots for training statistics
            if self.tensorboard:
                for key, value in self.stat_tracker.get_means().items():
                    self.tb_writer.add_scalar(key, value, iteration)

            # Get time of completion of current iteration
            iter_t = time.time()

            # Compute elapsed time for current iteration and append to `iter_times`
            self.iter_times = np.append(self.iter_times, iter_t - iter_t0)

            # Compute elapsed time so far
            elapsed_time = iter_t - t0

            # Compute rolling mean iteration time
            mean_iter_time = self.iter_times.mean()

            # Compute estimated time of experiment completion
            eta = elapsed_time * ((self.params.max_iter - iteration) / (iteration - starting_iter + 1))

            # Log progress in stdout
            if iteration % self.params.log_freq == 0:
                self.log_progress(iteration, mean_iter_time, elapsed_time, eta)

            # Save checkpoint model file and latent support_sets model state dicts after current iteration
            if iteration % self.params.ckp_freq == 0:
                # Build checkpoint dict
                checkpoint_dict = {
                    'iter': iteration,
                    'latent_support_sets': latent_support_sets.state_dict(),
                }
                torch.save(checkpoint_dict, self.checkpoint)
        # === End of training loop ===

        # Get experiment's total elapsed time
        elapsed_time = time.time() - t0

        # Save final latent support sets (LSS) model
        latent_support_sets_model_filename = osp.join(self.models_dir, 'latent_support_sets.pt')
        torch.save(latent_support_sets.state_dict(), latent_support_sets_model_filename)

        for _ in range(10):
            print()
        print("#.Training completed -- Total elapsed time: {}.".format(sec2dhms(elapsed_time)))

        print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
        try:
            # REVIEW: Do not ignore checkpoint; copy it to the complete model dir
            # shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
            shutil.copytree(src=self.wip_dir, dst=self.complete_dir)
            print("  \\__Done!")
        except IOError as e:
            print("  \\__Already exists -- {}".format(e))
