import torch
import numpy as np
import os.path as osp
import json
from sklearn import linear_model
from collections import defaultdict


class ExpPreprocess:
    def __init__(self, gan_type, gan_generator, stylegan_space, stylegan_layer, truncation, truncation_values=10,
                 num_samples_jung=10000, exp_dir=None, use_cuda=False, verbose=False):
        self.gan_type = gan_type
        self.gan_generator = gan_generator
        self.stylegan_space = stylegan_space
        self.stylegan_layer = stylegan_layer
        self.truncation = truncation
        self.truncation_values = truncation_values
        self.num_samples_jung = num_samples_jung
        self.exp_dir = exp_dir
        self.use_cuda = use_cuda
        self.verbose = verbose

        # TODO: add comment
        nested_dict = lambda: defaultdict(nested_dict)
        self.jung_radii_dict = nested_dict()
        self.jung_radii_file = osp.join('experiments', 'wip', self.exp_dir, 'jung_radii.json')

    def calculate_jung_radius(self):
        if self.verbose:
            print("#. Calculate Jung radius...")

        # If file already exists, read jung radius
        if osp.isfile(self.jung_radii_file):
            print("  \\__Jung radii file already exists -- get jung radius")

            with open(self.jung_radii_file, 'r') as f:
                jung_radii_dict = json.load(f)

            if 'stylegan' in self.gan_type:
                if self.stylegan_space == 'W+':
                    lm = jung_radii_dict[self.stylegan_space]['{}'.format(self.stylegan_layer)]
                else:
                    lm = jung_radii_dict[self.stylegan_space]
                jung_radius = lm[0] * self.truncation + lm[1]
            else:
                jung_radius = jung_radii_dict['Z'][1]

            return jung_radius

        ################################################################################################################
        ##                                                                                                            ##
        ##                                               [ StyleGANs ]                                                ##
        ##                                                                                                            ##
        ################################################################################################################
        if 'stylegan' in self.gan_type:
            ############################################################################################################
            ##                                       [ StyleGAN @ Z-space ]                                           ##
            ############################################################################################################
            if self.stylegan_space == 'Z':
                # Sample latent codes
                if self.verbose:
                    print("  \\__.{} (Z-space) using {} {}-dimensional samples...".format(
                        self.gan_type, self.num_samples_jung, self.gan_generator.dim_z))
                zs = torch.randn(self.num_samples_jung, self.gan_generator.dim_z)
                if self.use_cuda:
                    zs = zs.cuda()

                # TODO: add comment
                jung_radius = torch.cdist(zs, zs).max() * \
                    np.sqrt(self.gan_generator.dim_z / (2 * (self.gan_generator.dim_z + 1)))
                self.jung_radii_dict['Z'] = (0.0, jung_radius.cpu().detach().item())

            ############################################################################################################
            ##                                       [ StyleGAN @ W-space ]                                           ##
            ############################################################################################################
            elif self.stylegan_space == 'W':
                # Sample latent codes
                if self.verbose:
                    print("  \\__.{} W-space using {} {}-dimensional samples...".format(
                        self.gan_type, self.num_samples_jung, self.gan_generator.dim_z))
                zs = torch.randn(self.num_samples_jung, self.gan_generator.dim_z)
                if self.use_cuda:
                    zs = zs.cuda()

                # TODO: add comment
                data = []
                for truncation in np.linspace(0.1, 1.0, self.truncation_values):
                    with torch.no_grad():
                        ws = self.gan_generator.get_w(zs, truncation=truncation)[:, 0, :]
                    jung_radius = torch.cdist(ws, ws).max() * np.sqrt(ws.shape[1] / (2 * (ws.shape[1] + 1)))
                    data.append([truncation, jung_radius.cpu().detach().item()])
                data = np.array(data)

                lm = linear_model.LinearRegression()
                lm.fit(data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1))
                self.jung_radii_dict['W'] = (float(lm.coef_[0, 0]), float(lm.intercept_[0]))

            ############################################################################################################
            ##                                       [ StyleGAN @ W+-space ]                                          ##
            ############################################################################################################
            elif self.stylegan_space == 'W+':
                # Sample latent codes
                if self.verbose:
                    print("  \\__.{} W+-space using {} samples...".format(self.gan_type, self.num_samples_jung))
                zs = torch.randn(self.num_samples_jung, self.gan_generator.dim_z)
                if self.use_cuda:
                    zs = zs.cuda()

                # TODO: add comment
                data = []
                for truncation in np.linspace(0.1, 1.0, self.truncation_values):
                    with torch.no_grad():
                        ws_plus = self.gan_generator.get_w(zs, truncation=truncation)[:, :self.stylegan_layer + 1, :]
                    ws_plus = ws_plus.reshape(ws_plus.shape[0], -1)
                    jung_radius = torch.cdist(ws_plus, ws_plus).max() * \
                        np.sqrt(ws_plus.shape[1] / (2 * (ws_plus.shape[1] + 1)))
                    data.append([truncation, jung_radius.cpu().detach().item()])
                data = np.array(data)

                if self.verbose:
                    print("      \\__Fit linear model...")
                lm = linear_model.LinearRegression()
                lm.fit(data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1))
                self.jung_radii_dict['W+'][self.stylegan_layer] = (float(lm.coef_[0, 0]), float(lm.intercept_[0]))

            ############################################################################################################
            ##                                        [ StyleGAN @ S-space ]                                          ##
            ############################################################################################################
            elif self.stylegan_space == 'S':
                raise NotImplementedError
                # # Build GAN generator model and load with pre-trained weights
                # if args.verbose:
                #     print("  \\__Build GAN generator model G and load with pre-trained weights...")
                #     print("      \\__GAN generator : {} (res: {})".format(gan, GENFORCE_MODELS[gan][1]))
                #     print("      \\__Pre-trained weights: {}".format(GENFORCE_MODELS[gan][0]))
                #
                # G = load_generator(model_name=gan, latent_is_s=True).eval()
                #
                # # Upload GAN generator model to GPU
                # if use_cuda:
                #     G = G.cuda()
                #
                # # Latent codes sampling
                # if args.verbose:
                #     print("  \\__Sample {} {}-dimensional Z-space latent codes...".format(args.num_samples, G.dim_z))
                # zs = torch.randn(args.num_samples, G.dim_z)
                #
                # if use_cuda:
                #     zs = zs.cuda()
                #
                # # Calculate expected latent norm and fit a linear model for each version of the W+ space
                # if args.verbose:
                #     print("  \\__Calculate Jung radii and fit linear model in S-space...")
                #
                # data = []
                # for truncation in tqdm(np.linspace(0.1, 1.0, args.truncation_values),
                #                        desc="  \\__Calculate radii (S-space): "):
                #     with torch.no_grad():
                #         wp = G.get_w(zs, truncation=truncation)
                #
                #     wp_batches = torch.split(wp, args.batch_size)
                #     styles_dict_keys = list(G.get_s(wp[0, :, :].unsqueeze(0)).keys())
                #     d = dict()
                #     for k in styles_dict_keys:
                #         d.update({k: list()})
                #     styles_dict = dict.fromkeys(STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan].keys(), None)
                #     for t in range(len(wp_batches)):
                #         with torch.no_grad():
                #             styles_dict_t = G.get_s(wp_batches[t])
                #         for k, v in styles_dict_t.items():
                #             d[k].append(v)
                #     for k, v in d.items():
                #         if k in STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan].keys():
                #             styles_dict[k] = torch.cat(v)
                #
                #     target_styles_vector = torch.cat([styles_dict[k]
                #                                       for k in STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan].keys()], dim=1)
                #     jung_radius = torch.cdist(target_styles_vector, target_styles_vector).max() * \
                #                   np.sqrt(target_styles_vector.shape[1] / (2 * (target_styles_vector.shape[1] + 1)))
                #     data.append([truncation, jung_radius.cpu().detach().item()])
                #
                # data = np.array(data)
                # lm = linear_model.LinearRegression()
                # lm.fit(data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1))
                # jung_radii_dict[gan]['S'] = (float(lm.coef_[0, 0]), float(lm.intercept_[0]))

        ################################################################################################################
        ##                                                                                                            ##
        ##                                           [ ProgGAN @ Z-space]                                             ##
        ##                                                                                                            ##
        ################################################################################################################
        else:
            # Sample latent codes
            if self.verbose:
                print("  \\__{} (Z-space) using {} {}-dimensional samples...".format(
                    self.gan_type, self.num_samples_jung, self.gan_generator.dim_z))
            zs = torch.randn(self.num_samples_jung, self.gan_generator.dim_z)
            if self.use_cuda:
                zs = zs.cuda()

            # Calculate Jung radius
            jung_radius = torch.cdist(zs, zs).max() * \
                np.sqrt(self.gan_generator.dim_z / (2 * (self.gan_generator.dim_z + 1)))
            self.jung_radii_dict['Z'] = (0.0, jung_radius.cpu().detach().item())

        # Save jung radii file
        if self.verbose:
            print("  \\__Save Jung radi file...")
        with open(self.jung_radii_file, 'w') as fp:
            json.dump(self.jung_radii_dict, fp)

        # Return Jung radius
        with open(self.jung_radii_file, 'r') as f:
            jung_radii_dict = json.load(f)

        if 'stylegan' in self.gan_type:
            if self.stylegan_space == 'W+':
                lm = jung_radii_dict[self.stylegan_space]['{}'.format(self.stylegan_layer)]
            else:
                lm = jung_radii_dict[self.stylegan_space]
            jung_radius = lm[0] * self.truncation + lm[1]
        else:
            jung_radius = jung_radii_dict['Z'][1]

        return jung_radius