import sys
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

        nested_dict = lambda: defaultdict(nested_dict)
        self.jung_radii_dict = nested_dict()
        self.jung_radii_file = osp.join('experiments', 'wip', self.exp_dir, 'jung_radii.json')

    def calculate_latent_centres_and_jung_radii(self):
        """TODO: add description

        Note: In the case of StyleGAN2@W+ space, we calculate the latent centres and the Jung radii for all the desired
        layers even though they will be the same for synthetic codes. TODO: +++

        TODO: rename `jung_radii_dict` --> `latent_centres_jung_radii_dict`

        Returns:
            latent_centres (list) : TODO: +++
            jung_radii (list)     : TODO: +++
        """
        if self.verbose:
            print("#. Calculate latent centres and Jung radii...")

        latent_centres = []
        jung_radii = []

        # If file already exists, read latent centres and jung radii
        if osp.isfile(self.jung_radii_file):
            # TODO: rephrase...
            print("  \\__Latent centres and Jung radii file already exists -- get them!")

            with open(self.jung_radii_file, 'r') as f:
                jung_radii_dict = json.load(f)

            if 'stylegan' in self.gan_type:
                if self.stylegan_space == 'W+':
                    for layer_idx in range(self.stylegan_layer):
                        jung_radii.append(jung_radii_dict['W+']['{}'.format(layer_idx)]['jung_radius'])
                        latent_centres.append(torch.tensor(jung_radii_dict['W+']['{}'.format(layer_idx)]['latent_centre']))
                else:
                    jung_radii.append(jung_radii_dict['W']['jung_radius'])
                    latent_centres.append(torch.tensor(jung_radii_dict['W']['latent_centre']))

            else:
                jung_radii.append(jung_radii_dict['Z']['jung_radius'])
                latent_centres.append(torch.tensor(jung_radii_dict['Z']['latent_centre']))

            return latent_centres, jung_radii

        ################################################################################################################
        ##                                                                                                            ##
        ##                                               [ StyleGANs ]                                                ##
        ##                                                                                                            ##
        ################################################################################################################
        if 'stylegan' in self.gan_type:
            ############################################################################################################
            ##                                       [ StyleGAN @ W-space ]                                           ##
            ############################################################################################################
            if self.stylegan_space == 'W':
                # Sample latent codes
                if self.verbose:
                    print("  \\__.{} W-space using {} {}-dimensional samples...".format(
                        self.gan_type, self.num_samples_jung, self.gan_generator.dim_z))
                zs = torch.randn(self.num_samples_jung, self.gan_generator.dim_z)
                if self.use_cuda:
                    zs = zs.cuda()

                # TODO: add comment
                with torch.no_grad():
                    ws = self.gan_generator.get_w(zs, truncation=self.truncation)[:, 0, :]

                # Calculate latent centre
                latent_centre = torch.mean(ws, dim=0)

                # Calculate Jung radius
                jung_radius = torch.cdist(ws, ws).max() * np.sqrt(ws.shape[1] / (2 * (ws.shape[1] + 1)))

                # Update dict and return lists
                self.jung_radii_dict['W']['jung_radius'] = jung_radius.cpu().detach().item()
                self.jung_radii_dict['W']['latent_centre'] = latent_centre.cpu().detach().numpy().tolist()
                jung_radii.append(jung_radius)
                latent_centres.append(latent_centre)

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

                # Calculate the latent centre and the Jung radius of each W+ layer
                for layer_idx in range(self.stylegan_layer):
                    print("layer_idx: {}".format(layer_idx))

                    with torch.no_grad():
                        ws_plus = self.gan_generator.get_w(zs, truncation=self.truncation)[:, layer_idx, :]
                    ws_plus = ws_plus.reshape(ws_plus.shape[0], -1)

                    # Calculate latent centre
                    latent_centre = torch.mean(ws_plus, dim=0)

                    # Calculate Jung radius
                    jung_radius = torch.cdist(ws_plus, ws_plus).max() * \
                        np.sqrt(ws_plus.shape[1] / (2 * (ws_plus.shape[1] + 1)))

                    # Update dict and return lists
                    self.jung_radii_dict['W+'][layer_idx]['jung_radius'] = jung_radius.cpu().detach().item()
                    self.jung_radii_dict['W+'][layer_idx]['latent_centre'] = \
                        latent_centre.cpu().detach().numpy().tolist()
                    jung_radii.append(jung_radius)
                    latent_centres.append(latent_centre)

            ############################################################################################################
            ##                                        [ StyleGAN @ S-space ]                                          ##
            ############################################################################################################
            elif self.stylegan_space == 'S':
                raise NotImplementedError

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

            # Calculate latent centre
            latent_centre = torch.mean(zs, dim=0)

            # Calculate Jung radius
            jung_radius = torch.cdist(zs, zs).max() * \
                np.sqrt(self.gan_generator.dim_z / (2 * (self.gan_generator.dim_z + 1)))

            # Update dict and return lists
            self.jung_radii_dict['Z']['jung_radius'] = jung_radius.cpu().detach().item()
            self.jung_radii_dict['Z']['latent_centre'] = latent_centre.cpu().detach().numpy().tolist()
            jung_radii.append(jung_radius)
            latent_centres.append(latent_centre)

        # Save jung radii file
        if self.verbose:
            print("  \\__Save latent centres and Jung radi file...")
        with open(self.jung_radii_file, 'w') as fp:
            json.dump(self.jung_radii_dict, fp)

        return latent_centres, jung_radii
