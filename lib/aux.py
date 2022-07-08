import sys
import os
import os.path as osp
import json
import argparse
import numpy as np
import clip
import torch
import math
from scipy.stats import truncnorm
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage
from .config import SEMANTIC_DIPOLES_CORPORA


def create_exp_dir(args):
    """Create output directory for current experiment under experiments/wip/ and save given the arguments (json) and
    the given command (bash script).

    Experiment's directory name format:
        ContraCLIP-<gan_type>(-{Z,W,W+})-K<num_latent_support_sets>-D<num_latent_support_dipoles>-css_beta_<css_beta>
            -eps<min_shift_magnitude>_<max_shift_magnitude>
            (-<nonlinear_css_beta-<css_beta>/linear/styleclip>)(-<contrastive_<temperature>/cossim>)-<max_iter>-<prompt>

        E.g.:
            ContraCLIP_stylegan2_ffhq1024-W+-K3-D128-eps0.1_0.2-nonlinear_beta-0.75-contrastive_1.0-10000-expressions3

    Args:
        args (argparse.Namespace): the namespace object returned by `parse_args()` for the current run

    """
    exp_dir = "ContraCLIP_{}".format(args.gan)
    if 'stylegan' in args.gan:
        exp_dir += '-{}'.format(args.stylegan_space)
    else:
        exp_dir += '-Z'
    exp_dir += "-K{}-D{}".format(len(SEMANTIC_DIPOLES_CORPORA[args.corpus]), args.num_latent_support_dipoles)
    exp_dir += "-lss_beta_{}".format(args.lss_beta)
    exp_dir += "-eps{}_{}".format(args.min_shift_magnitude, args.max_shift_magnitude)
    if args.styleclip:
        exp_dir += "-styleclip"
    elif args.linear:
        exp_dir += "-linear"
    else:
        exp_dir += "-nonlinear_css_beta_{}".format(args.css_beta)

    exp_dir += "-{}".format(args.loss)
    if args.loss == "contrastive":
        exp_dir += "_{}".format(args.temperature)
    if args.id:
        exp_dir += "+{}xID".format(args.lambda_id)
    exp_dir += "-{}".format(args.max_iter)
    exp_dir += "-{}".format(args.corpus)

    # Create output directory (wip)
    wip_dir = osp.join("experiments", "wip", exp_dir)
    os.makedirs(wip_dir, exist_ok=True)
    # Save args namespace object in json format
    with open(osp.join(wip_dir, 'args.json'), 'w') as args_json_file:
        json.dump(args.__dict__, args_json_file)

    # Save the given command in a bash script file
    with open(osp.join(wip_dir, 'command.sh'), 'w') as command_file:
        command_file.write('#!/usr/bin/bash\n')
        command_file.write(' '.join(sys.argv) + '\n')

    return exp_dir


class PromptFeatures:
    def __init__(self, prompt_corpus, clip_model):
        self.prompt_corpus = prompt_corpus
        self.clip_model = clip_model.cpu()
        self.num_prompts = len(self.prompt_corpus)
        self.prompt_features_dim = 512

    def get_prompt_features(self):
        prompt_features = [
            self.clip_model.encode_text(clip.tokenize(self.prompt_corpus[t]).cpu()).unsqueeze(0) for t in
            range(len(self.prompt_corpus))
        ]
        return torch.cat(prompt_features, dim=0)


class TrainingStatTracker(object):
    def __init__(self):
        self.stat_tracker = {'loss': [], 'id_loss': []}

    def update(self, loss, id_loss):
        self.stat_tracker['loss'].append(float(loss))
        self.stat_tracker['id_loss'].append(float(id_loss))

    def get_means(self):
        stat_means = dict()
        for key, value in self.stat_tracker.items():
            stat_means.update({key: np.mean(value)})
        return stat_means

    def flush(self):
        for key in self.stat_tracker.keys():
            self.stat_tracker[key] = []


def sample_z(batch_size, dim_z, truncation=None):
    """Sample a random latent code from multi-variate standard Gaussian distribution with/without truncation.

    Args:
        batch_size (int)   : batch size (number of latent codes)
        dim_z (int)        : latent space dimensionality
        truncation (float) : truncation parameter

    Returns:
        z (torch.Tensor)   : batch of latent codes
    """
    if truncation is None or truncation == 1.0:
        return torch.randn(batch_size, dim_z)
    else:
        return torch.from_numpy(truncnorm.rvs(-truncation, truncation, size=(batch_size, dim_z))).to(torch.float)


def tensor2image(tensor, adaptive=False):
    tensor = tensor.squeeze(dim=0)
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


def update_progress(msg, total, progress):
    bar_length, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(bar_length * progress))
    block_symbol = u"\u2588"
    empty_symbol = u"\u2591"
    text = "\r{}{} {:.0f}% {}".format(msg, block_symbol * block + empty_symbol * (bar_length - block),
                                      round(progress * 100, 0), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def update_stdout(num_lines):
    """Update stdout by moving cursor up and erasing line for given number of lines.

    Args:
        num_lines (int): number of lines

    """
    cursor_up = '\x1b[1A'
    erase_line = '\x1b[1A'
    for _ in range(num_lines):
        print(cursor_up + erase_line)


def sec2dhms(t):
    """Convert time into days, hours, minutes, and seconds string format.

    Args:
        t (float): time in seconds

    Returns (string):
        "<days> days, <hours> hours, <minutes> minutes, and <seconds> seconds"

    """
    day = t // (24 * 3600)
    t = t % (24 * 3600)
    hour = t // 3600
    t %= 3600
    minutes = t // 60
    t %= 60
    seconds = t
    return "%02d days, %02d hours, %02d minutes, and %02d seconds" % (day, hour, minutes, seconds)


def get_wh(img_paths):
    """Get width and height of images in given list of paths. Images are expected to have the same resolution.

    Args:
        img_paths (list): list of image paths

    Returns:
        width (int)  : the common images width
        height (int) : the common images height

    """
    img_widths = []
    img_heights = []
    for img in img_paths:
        img_ = Image.open(img)
        img_widths.append(img_.width)
        img_heights.append(img_.height)

    if len(set(img_widths)) == len(set(img_heights)) == 1:
        return img_widths[0], img_heights[1]
    else:
        raise ValueError("Inconsistent image resolutions in {}".format(img_paths))


def create_summarizing_gif(imgs_root, gif_filename, num_imgs=None, gif_size=None, gif_fps=30, gap=15, progress_bar_h=15,
                           progress_bar_color=(252, 186, 3)):
    """Create a summarizing GIF image given an images root directory (images generated across a certain latent path) and
    the number of images to appear as a static sequence. The resolution of the resulting GIF image will be
    ((num_imgs + 1) * gif_size, gif_size). That is, a static sequence of `num_imgs` images will be depicted in front of
    the animated GIF image (the latter will use all the available images in `imgs_root`).

    Args:
        imgs_root (str)            : directory of images (generated across a certain path)
        gif_filename (str)         : filename of the resulting GIF image
        num_imgs (int)             : number of images that will be used to build the static sequence before the
                                     animated part of the GIF
        gif_size (int)             : height of the GIF image (its width will be equal to (num_imgs + 1) * gif_size)
        gif_fps (int)              : GIF frames per second
        gap (int)                  : a gap between the static sequence and the animated path of the GIF
        progress_bar_h (int)       : height of the progress bar depicted to the bottom of the animated part of the GIF
                                     image. If a non-positive number is given, progress bar will be disabled.
        progress_bar_color (tuple) : color of the progress bar

    """
    # Check if given images root directory exists
    if not osp.isdir(imgs_root):
        raise NotADirectoryError("Invalid directory: {}".format(imgs_root))

    # Get all images under given root directory
    path_images = [osp.join(imgs_root, dI) for dI in os.listdir(imgs_root) if osp.isfile(osp.join(imgs_root, dI))]
    path_images.sort()

    # Set number of images to appear in the static sequence of the GIF
    num_images = len(path_images)
    if num_imgs is None:
        num_imgs = num_images
    elif num_imgs > num_images:
        num_imgs = num_images

    # Get paths of static images
    static_imgs = []
    for i in range(0, len(path_images), math.ceil(len(path_images) / num_imgs)):
        static_imgs.append(osp.join(imgs_root, '{:06}.jpg'.format(i)))
    num_imgs = len(static_imgs)

    # Get GIF image resolution
    if gif_size is not None:
        gif_w = gif_h = gif_size
    else:
        gif_w, gif_h = get_wh(static_imgs)

    # Create PIL static image
    static_img_pil = Image.new('RGB', size=(len(static_imgs) * gif_w, gif_h))
    for i in range(len(static_imgs)):
        static_img_pil.paste(Image.open(static_imgs[i]).resize((gif_w, gif_h)), (i * gif_w, 0))

    # Create PIL GIF frames
    gif_frames = []
    for i in range(len(path_images)):
        # Create new PIL frame
        gif_frame_pil = Image.new('RGB', size=((num_imgs + 1) * gif_w + gap, gif_h), color=(255, 255, 255))

        # Paste static image
        gif_frame_pil.paste(static_img_pil, (0, 0))

        # Paste current image
        gif_frame_pil.paste(Image.open(path_images[i]).resize((gif_w, gif_h)), (num_imgs * gif_w + gap, 0))

        # Draw progress bar
        if progress_bar_h > 0:
            gif_frame_pil_drawing = ImageDraw.Draw(gif_frame_pil)
            progress = (i / len(path_images)) * gif_w
            gif_frame_pil_drawing.rectangle(xy=[num_imgs * gif_w + gap, gif_h - progress_bar_h,
                                                num_imgs * gif_w + gap + progress, gif_h],
                                            fill=progress_bar_color)

        # Append to GIF frames list
        gif_frames.append(gif_frame_pil)

    # Save GIF file
    gif_frames[0].save(
        fp=gif_filename,
        append_images=gif_frames[1:],
        save_all=True,
        optimize=False,
        loop=0,
        duration=1000 // gif_fps)
