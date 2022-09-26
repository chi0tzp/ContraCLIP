import sys
import os
import os.path as osp
import clip
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
import math
from torchvision import transforms
import numpy as np
import time
import shutil
from .aux import TrainingStatTracker, update_progress, update_stdout, sec2dhms
from .config import SEMANTIC_DIPOLES_CORPORA, STYLEGAN_LAYERS, STYLEGAN2_STYLE_SPACE_TARGET_LAYERS, GENFORCE_MODELS


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


pass


class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3,  # lr
                 alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
                 use_gc=True, gc_conv_only=False
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 ):

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.use_gc = use_gc

        # level of gradient centralization
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    # print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                state['step'] += 1

                # compute variance mov avg
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # exp_avg_sq.addcmul_()
                # REVIEW:
                #   Consider using one of the following signatures instead:
                # 	addcmul_(Tensor tensor1, Tensor tensor2, *, Number value)
                #   exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                #   gh:
                #   Try this:
                #   exp_avg_sq.mul_(beta2).addcmul(grad, grad, value = 1 - beta2)
                #   exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1 )

                exp_avg_sq.mul_(beta2).addcmul(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # compute mean moving avg
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # REVIEW:
                #   home/chi/LAB/ContraCLIP/lib/trainer.py:140: UserWarning: This overload of add_ is deprecated:
                # 	add_(Number alpha, Tensor other)
                #   Consider using one of the following signatures instead:
                # 	    add_(Tensor other, *, Number alpha) (Triggered internally at  ../torch/csrc/utils/python_arg_parser.cpp:1174.)
                #       exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    # REVIEW
                    # p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])

                else:
                    # REVIEW
                    # p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])




                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']  # get access to slow param tensor
                    # REVIEW
                    # slow_p.add_(self.alpha, p.data - slow_p)  # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)  # (fast weights - slow weights) * alpha

                    # exp_avg.mul_(beta1).add_(1 - beta1, grad) -->
                    # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                    p.data.copy_(slow_p)  # copy interpolated weights to RAdam param tensor

        return loss




class Trainer(object):
    def __init__(self, params=None, exp_dir=None, exp_subdir=None, use_cuda=False, multi_gpu=False):
        if params is None:
            raise ValueError("Cannot build a Trainer instance with empty params: params={}".format(params))
        else:
            self.params = params
        self.exp_dir = exp_dir
        self.exp_subdir = exp_subdir
        self.use_cuda = use_cuda
        self.multi_gpu = multi_gpu

        # Set output directory for current experiment (wip)
        self.wip_dir = osp.join("experiments", "wip", self.exp_dir)

        # Set directory for completed experiment
        if exp_subdir:
            self.complete_dir = osp.join("experiments", "complete", self.exp_subdir, self.exp_dir)
        else:
            self.complete_dir = osp.join("experiments", "complete", self.exp_dir)

        # Create log subdirectory and define stat.json file
        self.stats_json = osp.join(self.wip_dir, 'stats.json')
        if not osp.isfile(self.stats_json):
            with open(self.stats_json, 'w') as out:
                json.dump({}, out)

        # Create models sub-directory
        self.models_dir = osp.join(self.wip_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        # Define checkpoint model file
        self.checkpoint = osp.join(self.models_dir, 'checkpoint.pt')

        # Array of iteration times
        self.iter_times = np.array([])

        # Set up training statistics tracker
        self.stat_tracker = TrainingStatTracker()

        # REVIEW: StyleCLIP's CLIP image approach
        self.upsample = torch.nn.Upsample(scale_factor=7)
        # self.avg_pool = torch.nn.AvgPool2d(kernel_size=GENFORCE_MODELS[self.params.gan][1] // 32)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=GENFORCE_MODELS[self.params.gan][1] // 16)

        # Define transform of CLIP image encoder
        self.clip_img_transform = transforms.Compose([transforms.Resize(224),
                                                      transforms.CenterCrop(224),
                                                      transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                           (0.26862954, 0.26130258, 0.27577711))])

        # Define cross entropy loss
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # Define cosine similarity loss
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss()

    def contrastive_loss(self, img_batch, txt_batch):
        n_img, d_img = img_batch.shape

        # Normalise image and text batches
        img_batch_l2 = F.normalize(img_batch, p=2, dim=-1)
        txt_batch_l2 = F.normalize(txt_batch, p=2, dim=-1)

        # Calculate inner product similarity matrix
        similarity_matrix = torch.matmul(img_batch_l2, txt_batch_l2.T)
        labels = torch.arange(n_img)

        # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # REVIEW:
        # return self.cross_entropy_loss(similarity_matrix / self.params.temperature, labels) +\
        #     self.cross_entropy_loss(similarity_matrix.T / self.params.temperature, labels)

        return self.cross_entropy_loss(similarity_matrix / self.params.temperature, labels)

    def get_starting_iteration(self, latent_support_sets, corpus_support_sets):
        """Check if checkpoint file exists (under `self.models_dir`) and set starting iteration at the checkpoint
        iteration; also load checkpoint weights to `latent_support_sets` and `corpus_support_sets`. Otherwise, set
        starting iteration to 1 in order to train from scratch.

        Returns:
            starting_iter (int): starting iteration

        """
        starting_iter = 1
        if osp.isfile(self.checkpoint):
            checkpoint_dict = torch.load(self.checkpoint)
            starting_iter = checkpoint_dict['iter']
            latent_support_sets.load_state_dict(checkpoint_dict['latent_support_sets'])
            corpus_support_sets.load_state_dict(checkpoint_dict['corpus_support_sets'])

        return starting_iter

    def log_progress(self, iteration, mean_iter_time, elapsed_time, eta, last_lr):
        """Log progress (loss + ETA).

        Args:
            iteration (int)        : current iteration
            mean_iter_time (float) : mean iteration time
            elapsed_time (float)   : elapsed time until current iteration
            eta (float)            : estimated time of experiment completion
            last_lr (list)         : last learning rates
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

        last_lr = ', '.join('{:.6f}'.format(lr) for lr in last_lr)
        update_progress("  \\__.Training [bs: {}] [lr: {}] [iter: {:06d}/{:06d}] ".format(
            self.params.batch_size, last_lr, iteration, self.params.max_iter), self.params.max_iter, iteration + 1)
        if iteration < self.params.max_iter - 1:
            print()
        print("         ===================================================================")
        print("      \\__Loss    : {:.04f} (tau={:.02f})".format(stats['loss'], self.params.temperature))
        if self.params.id:
            print("      \\__Loss ID : {:.04f} (lambda={:.02f})".format(stats['loss_id'], self.params.lambda_id))
        print("         ===================================================================")
        print("      \\__Mean iter time : {:.3f} sec".format(mean_iter_time))
        print("      \\__Elapsed time   : {}".format(sec2dhms(elapsed_time)))
        print("      \\__ETA            : {}".format(sec2dhms(eta)))
        print("         ===================================================================")

        if self.params.id:
            update_stdout(9)
        else:
            update_stdout(8)

    def train(self, generator, latent_support_sets, corpus_support_sets, clip_model, id_loss=None):
        """ContraCLIP training function.

        Args:
            generator           : non-trainable (pre-trained) GAN generator
            latent_support_sets : trainable LSS model -- interpretable latent paths model
            corpus_support_sets : CSS model -- non-linear paths in the CLIP space (trainable or non-trainable based on
                                  `self.params.id`)
            clip_model          : non-trainable (pre-trained) CLIP model
            id_loss             : ID preserving criterion (ArcFace)
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
            latent_support_sets.cuda().train()
            corpus_support_sets.cuda().eval()
            if self.params.id:
                id_loss.cuda().eval()
        else:
            generator.eval()
            clip_model.eval()
            latent_support_sets.train()
            corpus_support_sets.eval()
            if self.params.id:
                id_loss.eval()

        # Set up optimizer
        # TODO:
        #   - Check `amsgrad=True`
        optimizer = torch.optim.Adam(params=latent_support_sets.parameters(), lr=self.params.lr)
        # optimizer = Ranger(params=latent_support_sets.parameters(), lr=self.params.lr)

        # Set learning rate scheduler
        lr_scheduler = MultiStepLR(optimizer=optimizer,
                                   milestones=[int(0.60 * self.params.max_iter),
                                               int(0.80 * self.params.max_iter),
                                               int(0.95 * self.params.max_iter)],
                                   gamma=0.1)

        # Get starting iteration
        starting_iter = self.get_starting_iteration(latent_support_sets, corpus_support_sets)

        # Parallelize models into multiple GPUs, if available and `multi_gpu=True`.
        if self.multi_gpu:
            print("#. Parallelize G and CLIP over {} GPUs...".format(torch.cuda.device_count()))
            # Parallelize generator G
            generator = DataParallelPassthrough(generator)
            # Parallelize CLIP model
            clip_model = DataParallelPassthrough(clip_model)

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

        ############################################################################################################
        ##                                                                                                        ##
        ##                                          [ Training Loop ]                                             ##
        ##                                                                                                        ##
        ############################################################################################################
        for iteration in range(starting_iter, self.params.max_iter + 1):
            # Get current iteration's start time
            iter_t0 = time.time()

            # Set gradients to zero
            generator.zero_grad()
            latent_support_sets.zero_grad()
            clip_model.zero_grad()
            if self.params.id:
                id_loss.zero_grad()

            # Sample latent code from standard Gaussian
            z = torch.randn(1, generator.dim_z)

            if self.use_cuda:
                z = z.cuda()

            ############################################################################################################
            ##                          [ Generate images using the original latent codes ]                           ##
            ############################################################################################################
            latent_code = z
            if 'stylegan' in self.params.gan:
                if self.params.stylegan_space == 'W':
                    latent_code = generator.get_w(z, truncation=self.params.truncation)[:, 0, :]
                elif self.params.stylegan_space == 'W+':
                    latent_code = generator.get_w(z, truncation=self.params.truncation)
                elif self.params.stylegan_space == 'S':
                    latent_code = generator.get_s(generator.get_w(z, truncation=self.params.truncation))
            img = generator(latent_code)
            img = img.repeat(self.params.batch_size, 1, 1, 1)

            # Repeat latent code
            if len(latent_code.shape) == 2:
                latent_code = latent_code.repeat(self.params.batch_size, 1)
            elif len(latent_code.shape) == 3:
                latent_code = latent_code.repeat(self.params.batch_size, 1, 1)
            else:
                raise ValueError("Invalid latent code shape.")

            ############################################################################################################
            ##                                   [ Calculate latent shift vectors ]                                   ##
            ############################################################################################################

            # Sample indices of shift vectors (`self.params.batch_size` out of `self.params.num_support_sets`)
            # target_support_sets_indices = torch.randint(0, self.params.num_support_sets, [self.params.batch_size])
            # REVIEW: previous buggy implementation of indices sampling
            # target_support_sets_indices = torch.randint(0, latent_support_sets.num_support_sets,
            #                                             [self.params.batch_size])

            tmp_ids = torch.ones(latent_support_sets.num_support_sets).multinomial(num_samples=self.params.batch_size,
                                                                                   replacement=False)
            target_support_sets_indices = torch.arange(latent_support_sets.num_support_sets)[tmp_ids]
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
            shift_magnitudes_pool = shift_magnitudes_pool[torch.randperm(shift_magnitudes_pool.shape[0])]
            shift_magnitudes_ids = torch.arange(len(shift_magnitudes_pool), dtype=torch.float)
            target_shift_magnitudes = shift_magnitudes_pool[torch.multinomial(input=shift_magnitudes_ids,
                                                                              num_samples=self.params.batch_size,
                                                                              replacement=False)]
            if self.use_cuda:
                target_shift_magnitudes = target_shift_magnitudes.cuda()


            # Create support sets mask of size (batch_size, num_support_sets) in the form:
            #       support_sets_mask[i] = [0, ..., 0, 1, 0, ..., 0]
            # where 1 indicates the i-th element of `target_support_sets_indices`
            support_sets_mask = torch.zeros([self.params.batch_size, latent_support_sets.num_support_sets])
            target_shift_signs = torch.zeros_like(target_shift_magnitudes)
            prompt_mask = torch.zeros([self.params.batch_size, 2])
            for i, (index, val) in enumerate(zip(target_support_sets_indices, target_shift_magnitudes)):
                support_sets_mask[i][index] += 1.0
                # target_shift_signs[i] = +1.0 if val >= 0 else -1.0
                if val >= 0:
                    prompt_mask[i, 0] = 1.0
                    target_shift_signs[i] = 1.0
                else:
                    prompt_mask[i, 1] = 1.0
                    target_shift_signs[i] = -1.0
            prompt_mask = prompt_mask.unsqueeze(1)
            if self.use_cuda:
                support_sets_mask = support_sets_mask.cuda()
                prompt_mask = prompt_mask.cuda()
                target_shift_signs = target_shift_signs.cuda()

            # Calculate shift vectors for the given latent codes -- in the case of StyleGAN, shifts live in the
            # self.params.stylegan_space, i.e., in Z-, W-, W+, S space. In the Z/W-space the dimensionality of the
            # latent space is 512, in the W+-space the dimensionality is 512 * (self.params.stylegan_layer + 1), and in
            # the S-space the dimensionality of the latent space is
            # sum(STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[self.params.gan])
            latent_shift = None
            if ('stylegan' in self.params.gan) and (self.params.stylegan_space in ('W+', 'S')):
                if self.params.stylegan_space == 'W+':
                    latent_shift = target_shift_magnitudes.reshape(-1, 1) * latent_support_sets(
                        support_sets_mask,
                        latent_code[:, :self.params.stylegan_layer + 1, :].reshape(latent_code.shape[0], -1))

                    # print("target_shift_magnitudes : {}".format(target_shift_magnitudes))
                    # print("latent_shift            : {}".format(latent_shift.shape))
                    # print(torch.norm(latent_shift, dim=1))
                    # sys.exit()


                elif self.params.stylegan_space == 'S':
                    latent_shift = target_shift_magnitudes.reshape(-1, 1) * latent_support_sets(
                        support_sets_mask,
                        torch.cat([latent_code[k]
                                   for k in STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[self.params.gan].keys()], dim=1))
            else:
                latent_shift = target_shift_magnitudes.reshape(-1, 1) * latent_support_sets(support_sets_mask,
                                                                                            latent_code)

            ############################################################################################################
            ##               [ Add latent shift vectors to original latent codes and generate images ]                ##
            ############################################################################################################
            img_shifted = None
            if ('stylegan' in self.params.gan) and (self.params.stylegan_space in ('W+', 'S')):
                if self.params.stylegan_space == 'W+':
                    latent_code_reshaped = latent_code.reshape(latent_code.shape[0], -1)
                    latent_shift = F.pad(
                        input=latent_shift,
                        pad=(0, (STYLEGAN_LAYERS[self.params.gan] - 1 - self.params.stylegan_layer) * 512),
                        mode='constant',
                        value=0)
                    latent_code_shifted = latent_code_reshaped + latent_shift
                    latent_code_shifted_reshaped = latent_code_shifted.reshape_as(latent_code)
                    img_shifted = generator(latent_code_shifted_reshaped)
                elif self.params.stylegan_space == 'S':
                    latent_code_target_styles_vector = torch.cat(
                        [latent_code[k] for k in STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[self.params.gan].keys()], dim=1)
                    latent_code_target_styles_vector = latent_code_target_styles_vector + latent_shift
                    latent_code_target_styles_tuple = \
                        torch.split(
                            tensor=latent_code_target_styles_vector,
                            split_size_or_sections=list(STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[self.params.gan].values()),
                            dim=1)
                    latent_code_shifted = dict()
                    cnt = 0
                    for k, v in latent_code.items():
                        if k in STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[self.params.gan]:
                            latent_code_shifted.update({k: latent_code_target_styles_tuple[cnt]})
                            cnt += 1
                        else:
                            latent_code_shifted.update({k: latent_code[k]})
                    img_shifted = generator(latent_code_shifted)
            else:
                img_shifted = generator(latent_code + latent_shift)

            ############################################################################################################
            ##                                  [ Image features on the VL sphere ]                                   ##
            ############################################################################################################
            # Get VL (CLIP) image features for the original and the manipulated (shifted) images

            # REVIEW: check StyleCLIP's approach....
            # vl_img_pairs = clip_model.encode_image(self.clip_img_transform(torch.cat([img, img_shifted], dim=0)))
            # print(self.clip_img_transform((self.avg_pool(self.upsample(torch.cat([img, img_shifted], dim=0))))).shape)
            # print(self.avg_pool(self.upsample(torch.cat([img, img_shifted], dim=0))).shape)
            # sys.exit()
            vl_img_pairs = clip_model.encode_image(self.avg_pool(self.upsample(torch.cat([img, img_shifted], dim=0))))
            vl_img, vl_img_shifted = torch.split(vl_img_pairs, img.shape[0], dim=0)

            # Normalise image features for the original and the manipulated (shifted) images
            # (i.e., project of the VL sphere)
            vl_img = F.normalize(vl_img, p=2, dim=-1)
            vl_img_shifted = F.normalize(vl_img_shifted, p=2, dim=-1)

            ############################################################################################################
            ##                                   [  ]                                    ##
            ############################################################################################################
            loss = None
            loss_id = 0.0
            loss_w = 0.0
            # if self.params.vl_sim == "proposed-warping":
            #     # Calculate the orthogonal projection of the difference of the VL image features, i.e., the manipulated
            #     # (vl_img_shifted) minus the original (vl_img) onto the tangent space T_{vl_img}S^{n-1}
            #     vl_img_diff = corpus_support_sets.orthogonal_projection(s=vl_img.float(),
            #                                                             w=(vl_img_shifted - vl_img).float())
            #
            #     # TODO: add comment
            #     vl_txt = target_shift_signs.reshape(-1, 1) * corpus_support_sets(support_sets_mask, vl_img)
            #
            #     # Contrastive loss
            #     loss = self.contrastive_loss(vl_img_diff, vl_txt)

            if self.params.vl_sim == "proposed-warping":

                semantic_dipoles_features_cls_batch = torch.matmul(
                    support_sets_mask, corpus_support_sets.SEMANTIC_DIPOLES_FEATURES_CLS).reshape(
                    -1, 2, corpus_support_sets.support_vectors_dim)
                pole_vectors = torch.matmul(prompt_mask, semantic_dipoles_features_cls_batch).squeeze(1)

                vl_img_diff = corpus_support_sets.orthogonal_projection(s=pole_vectors,
                                                                        w=(vl_img_shifted - vl_img).float())


                vl_txt = target_shift_signs.reshape(-1, 1) * corpus_support_sets(support_sets_mask, vl_img)

                # Contrastive loss
                loss = self.contrastive_loss(vl_img_diff, vl_txt)


            ############################################################################################################
            ##                                   [  ]                                    ##
            ############################################################################################################
            elif self.params.vl_sim == "proposed-warping-aux":
                # Calculate the orthogonal projection of the difference of the VL image features, i.e., the manipulated
                # (vl_img_shifted) minus the original (vl_img) onto the tangent space T_{vl_img}S^{n-1}
                vl_img_diff = corpus_support_sets.orthogonal_projection(s=vl_img.float(),
                                                                        w=(vl_img_shifted - vl_img).float())

                # Get the vectors between the ending and the starting poles
                semantic_dipoles_features_cls_batch = torch.matmul(
                    support_sets_mask, corpus_support_sets.SEMANTIC_DIPOLES_FEATURES_CLS).reshape(
                    -1, 2, corpus_support_sets.support_vectors_dim)

                # Calculate the differences
                pole_vectors_diff = target_shift_signs.reshape(-1, 1) * \
                    (semantic_dipoles_features_cls_batch[:, 0, :] - semantic_dipoles_features_cls_batch[:, 1, :])

                # Calculate the orthogonal projection of the pole difference features, i.e., the ending minus the
                # starting pole onto the tangent space T_{vl_img}S^{n-1}
                vl_txt = corpus_support_sets.orthogonal_projection(s=vl_img.float(), w=pole_vectors_diff)

                # REVIEW
                vl_txt_warp = target_shift_signs.reshape(-1, 1) * corpus_support_sets(support_sets_mask, vl_img)
                lambda_w = 1.0

                # print()
                # print("lambda_w = {}".format(lambda_w))
                # print("\tloss w/o  warping : {:.04f}".format(self.contrastive_loss(vl_img_diff, vl_txt)))
                # print("\tloss WITH warping : {:.04f}".format(self.contrastive_loss(vl_img_diff,  + lambda_w * vl_txt_warp)))
                # print('---')
                # print()

                # input('__>')
                # sys.exit()

                # Contrastive loss
                loss = self.contrastive_loss(vl_img_diff, vl_txt + lambda_w * vl_txt_warp)

                # loss_w = self.contrastive_loss(vl_img_diff, vl_txt + lambda_w * vl_txt_warp)

                # TODO:
                #   - save loss with and wothout warping rectif...

            ############################################################################################################
            ##                             [ Proposed Image-Text Similarity (No-warping) ]                            ##
            ############################################################################################################
            elif self.params.vl_sim == "proposed-no-warping":
                # Calculate the orthogonal projection of the difference of the VL image features, i.e., the manipulated
                # (vl_img_shifted) minus the original (vl_img) onto the tangent space T_{vl_img}S^{n-1}
                vl_img_diff = corpus_support_sets.orthogonal_projection(s=vl_img.float(),
                                                                        w=(vl_img_shifted - vl_img).float())

                # Get the vectors between the ending and the starting poles
                semantic_dipoles_features_cls_batch = torch.matmul(
                    support_sets_mask, corpus_support_sets.SEMANTIC_DIPOLES_FEATURES_CLS).reshape(
                    -1, 2, corpus_support_sets.support_vectors_dim)

                # Calculate the differences
                pole_vectors_diff = target_shift_signs.reshape(-1, 1) * \
                    (semantic_dipoles_features_cls_batch[:, 0, :] - semantic_dipoles_features_cls_batch[:, 1, :])

                # Calculate the orthogonal projection of the pole difference features, i.e., the ending minus the
                # starting pole onto the tangent space T_{vl_img}S^{n-1}
                # REVIEW: No orthogonal projection
                vl_txt = corpus_support_sets.orthogonal_projection(s=vl_img.float(), w=pole_vectors_diff)

                # Contrastive loss
                loss = self.contrastive_loss(vl_img_diff, vl_txt)

            ############################################################################################################
            ##                                   [ Standard Image-Text Similarity ]                                   ##
            ############################################################################################################
            elif self.params.vl_sim == "standard":
                # Get the corresponding pole vectors
                semantic_dipoles_features_cls_batch = torch.matmul(
                    support_sets_mask, corpus_support_sets.SEMANTIC_DIPOLES_FEATURES_CLS).reshape(
                    -1, 2, corpus_support_sets.support_vectors_dim)
                pole_vectors = torch.matmul(prompt_mask, semantic_dipoles_features_cls_batch).squeeze(1)

                # Contrastive loss
                loss = self.contrastive_loss(vl_img_shifted.float(), pole_vectors)
            ############################################################################################################

            # Calculate ID preserving loss (ArcFace) in the case of face-generating GAN (if self.params.id is set)
            if self.params.id:
                loss_id = self.params.lambda_id * id_loss(y_hat=img_shifted, y=img)
                loss += loss_id

            # Update statistics tracker
            self.stat_tracker.update(loss=loss.item(), loss_id=loss_id, loss_w=loss_w)

            # Back-propagate
            loss.backward()

            # Update weights
            clip_model.float()
            optimizer.step()
            lr_scheduler.step()
            clip.model.convert_weights(clip_model)
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
                self.log_progress(iteration=iteration,
                                  mean_iter_time=mean_iter_time,
                                  elapsed_time=elapsed_time,
                                  eta=eta,
                                  last_lr=lr_scheduler.get_last_lr())

            # Save checkpoint model file and latent support_sets model state dicts after current iteration
            if iteration % self.params.ckp_freq == 0:
                # Build checkpoint dict
                checkpoint_dict = {
                    'iter': iteration,
                    'latent_support_sets': latent_support_sets.state_dict(),
                    'corpus_support_sets': corpus_support_sets.state_dict()
                }
                torch.save(checkpoint_dict, self.checkpoint)
        # === End of training loop ===

        # Get experiment's total elapsed time
        elapsed_time = time.time() - t0

        # Save final latent and corpus support sets models
        latent_support_sets_model_filename = osp.join(self.models_dir, 'latent_support_sets.pt')
        torch.save(latent_support_sets.state_dict(), latent_support_sets_model_filename)
        corpus_support_sets_model_filename = osp.join(self.models_dir, 'corpus_support_sets.pt')
        torch.save(corpus_support_sets.state_dict(), corpus_support_sets_model_filename)

        for _ in range(10):
            print()
        print("#.Training completed -- Total elapsed time: {}.".format(sec2dhms(elapsed_time)))

        print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
        try:
            shutil.copytree(src=self.wip_dir, dst=self.complete_dir)
            print("  \\__Done!")
        except IOError as e:
            print("  \\__Already exists -- {}".format(e))
