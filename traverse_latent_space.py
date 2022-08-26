import sys
import argparse
import os
import os.path as osp
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image, ImageDraw
import json
from torchvision.transforms import ToPILImage
from lib import LatentSupportSets, GENFORCE_MODELS, update_progress, update_stdout, STYLEGAN_LAYERS, \
    STYLEGAN2_STYLE_SPACE_TARGET_LAYERS
from models.load_generator import load_generator


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class ModelArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def tensor2image(tensor, img_size=None, adaptive=False):
    # Squeeze tensor image
    tensor = tensor.squeeze(dim=0)
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        if img_size:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8)).resize((img_size, img_size))
        else:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        if img_size:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8)).resize((img_size, img_size))
        else:
            return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


def one_hot(dims, value, idx):
    vec = torch.zeros(dims)
    vec[idx] = value
    return vec


def create_strip(image_list, N=5, strip_height=256):
    """Create strip of images across a given latent path.

    Args:
        image_list (list)  : list of images (PIL.Image.Image) across a given latent path
        N (int)            : number of images in strip
        strip_height (int) : strip height in pixels -- its width will be N * strip_height


    Returns:
        transformed_images_strip (PIL.Image.Image) : strip PIL image
    """
    step = len(image_list) // N + 1
    transformed_images_strip = Image.new('RGB', (N * strip_height, strip_height))
    for i in range(N):
        j = i * step if i * step < len(image_list) else len(image_list) - 1
        transformed_images_strip.paste(image_list[j].resize((strip_height, strip_height)), (i * strip_height, 0))
    return transformed_images_strip


def create_gif(image_list, gif_height=256):
    """Create gif frames for images across a given latent path.

    Args:
        image_list (list) : list of images (PIL.Image.Image) across a given latent path
        gif_height (int)  : gif height in pixels -- its width will be N * gif_height

    Returns:
        transformed_images_gif_frames (list): list of gif frames in PIL (PIL.Image.Image)
    """
    transformed_images_gif_frames = []
    for i in range(len(image_list)):
        # Create gif frame
        gif_frame = Image.new('RGB', (2 * gif_height, gif_height))
        gif_frame.paste(image_list[len(image_list) // 2].resize((gif_height, gif_height)), (0, 0))
        gif_frame.paste(image_list[i].resize((gif_height, gif_height)), (gif_height, 0))

        # Draw progress bar
        draw_bar = ImageDraw.Draw(gif_frame)
        bar_h = 12
        bar_colour = (252, 186, 3)
        draw_bar.rectangle(xy=((gif_height, gif_height - bar_h),
                               ((1 + (i / len(image_list))) * gif_height, gif_height)),
                           fill=bar_colour)

        transformed_images_gif_frames.append(gif_frame)

    return transformed_images_gif_frames


def main():
    """ContraCLIP -- Latent space traversal script.

    A script for traversing the latent space of a pre-trained GAN generator through paths defined by the warpings of
    a set of pre-trained support vectors. Latent codes are drawn from a pre-defined collection via the `--pool`
    argument. The generated images are stored under `results/` directory.

    Options:
        ================================================================================================================
        -v, --verbose : set verbose mode on
        ================================================================================================================
        --exp         : set experiment's model dir, as created by `train.py`, i.e., it should contain a subdirectory
                        `models/` with two files, namely `reconstructor.pt` and `support_sets.pt`, which
                        contain the weights for the reconstructor and the support sets, respectively, and an `args.json`
                        file that contains the arguments the model has been trained with.
        --pool        : directory of pre-defined pool of latent codes (created by `sample_gan.py`)
        --w-space     : latent codes in the pool are in W/W+ space (typically as inverted codes of real images)
        ================================================================================================================
        --shift-steps : set number of shifts to be applied to each latent code at each direction (positive/negative).
                        That is, the total number of shifts applied to each latent code will be equal to
                        2 * args.shift_steps.
        --eps         : set shift step magnitude for generating G(z'), where z' = z +/- eps * direction.
        --shift-leap  : set path shift leap (after how many steps to generate images)
        --batch-size  : set generator batch size (if not set, use the total number of images per path)
        --img-size    : set size of saved generated images (if not set, use the output size of the respective GAN
                        generator)
        --img-quality : JPEG image quality (max 95)
        --gif         : generate collated GIF images for all paths and all latent codes
        --gif-height  : set GIF image height -- width will be 2 * args.gif_height
        --gif-fps     : set number of frames per second for the generated GIF images
        --strip       : create traversal strip images
        --strip-number : set number of images per strip
        --strip-height : set strip height -- width will be 2 * args.strip_height
        ================================================================================================================
        --cuda        : use CUDA (default)
        --no-cuda     : do not use CUDA
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="ContraCLIP latent space traversal script")
    parser.add_argument('-v', '--verbose', action='store_true', help="set verbose mode on")
    # ================================================================================================================ #
    parser.add_argument('--w-space', action='store_true', help="latent codes are given in the W-space")
    parser.add_argument('--exp', type=str, required=True, help="set experiment's model dir (created by `train.py`)")
    parser.add_argument('--pool', type=str, required=True, help="directory of pre-defined pool of latent codes"
                                                                "(created by `sample_gan.py`)")
    parser.add_argument('--shift-steps', type=int, default=16, help="set number of shifts per positive/negative path "
                                                                    "direction")
    parser.add_argument('--eps', type=float, default=0.2, help="set shift step magnitude")
    parser.add_argument('--shift-leap', type=int, default=1,
                        help="set path shift leap (after how many steps to generate images)")
    parser.add_argument('--batch-size', type=int, help="set generator batch size (if not set, use the total number of "
                                                       "images per path)")
    parser.add_argument('--img-size', type=int, help="set size of saved generated images (if not set, use the output "
                                                     "size of the respective GAN generator)")
    parser.add_argument('--img-quality', type=int, default=50, help="set JPEG image quality")

    parser.add_argument('--strip', action='store_true', help="create traversal strip images")
    parser.add_argument('--strip-number', type=int, default=9, help="set number of images per strip")
    parser.add_argument('--strip-height', type=int, default=256, help="set strip height")
    parser.add_argument('--gif', action='store_true', help="create GIF traversals")
    parser.add_argument('--gif-height', type=int, default=256, help="set gif height")
    parser.add_argument('--gif-fps', type=int, default=30, help="set gif frame rate")
    # ================================================================================================================ #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Check structure of `args.exp`
    if not osp.isdir(args.exp):
        raise NotADirectoryError("Invalid given directory: {}".format(args.exp))

    # -- args.json file (pre-trained model arguments)
    args_json_file = osp.join(args.exp, 'args.json')
    if not osp.isfile(args_json_file):
        raise FileNotFoundError("File not found: {}".format(args_json_file))
    args_json = ModelArgs(**json.load(open(args_json_file)))
    gan = args_json.__dict__["gan"]
    stylegan_space = args_json.__dict__["stylegan_space"]
    stylegan_layer = args_json.__dict__["stylegan_layer"] if "stylegan_layer" in args_json.__dict__ else None
    truncation = args_json.__dict__["truncation"]
    learn_gammas = args_json.__dict__["learn_gammas"]

    # TODO: Check if `--w-space` is valid
    if args.w_space and (('stylegan' not in gan) or ('W' not in stylegan_space)):
        raise NotImplementedError

    # -- models directory (support sets and reconstructor, final or checkpoint files)
    models_dir = osp.join(args.exp, 'models')
    if not osp.isdir(models_dir):
        raise NotADirectoryError("Invalid models directory: {}".format(models_dir))

    # ---- Get all files of models directory
    models_dir_files = [f for f in os.listdir(models_dir) if osp.isfile(osp.join(models_dir, f))]

    # ---- Check for latent support sets (LSS) model file (final or checkpoint)
    latent_support_sets_model = osp.join(models_dir, 'latent_support_sets.pt')
    model_iter = ''
    if not osp.isfile(latent_support_sets_model):
        latent_support_sets_checkpoint_files = []
        for f in models_dir_files:
            if 'latent_support_sets-' in f:
                latent_support_sets_checkpoint_files.append(f)
        latent_support_sets_checkpoint_files.sort()
        latent_support_sets_model = osp.join(models_dir, latent_support_sets_checkpoint_files[-1])
        model_iter = '-{}'.format(latent_support_sets_checkpoint_files[-1].split('.')[0].split('-')[-1])

    # -- Get prompt corpus list
    with open(osp.join(models_dir, 'semantic_dipoles.json'), 'r') as f:
        semantic_dipoles = json.load(f)

    # Check given pool directory
    pool = osp.join('experiments', 'latent_codes', gan, args.pool)
    if not osp.isdir(pool):
        raise NotADirectoryError("Invalid pool directory: {} -- Please run sample_gan.py to create it.".format(pool))

    # CUDA
    use_cuda = False
    multi_gpu = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if torch.cuda.device_count() > 1:
                multi_gpu = True
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Build GAN generator model and load with pre-trained weights
    if args.verbose:
        print("#. Build GAN generator model G and load with pre-trained weights...")
        print("  \\__GAN generator : {} (res: {})".format(gan, GENFORCE_MODELS[gan][1]))
        print("  \\__Pre-trained weights: {}".format(GENFORCE_MODELS[gan][0]))

    G = load_generator(model_name=gan,
                       latent_is_w=('stylegan' in gan) and ('W' in args_json.__dict__["stylegan_space"]),
                       latent_is_s=('stylegan' in gan) and args_json.__dict__["stylegan_space"] == 'S',
                       verbose=args.verbose).eval()

    # Upload GAN generator model to GPU
    if use_cuda:
        G = G.cuda()

    # Parallelize GAN generator model into multiple GPUs if available
    if multi_gpu:
        G = DataParallelPassthrough(G)

    # Build latent support sets model LSS
    if args.verbose:
        print("#. Build Latent Support Sets model LSS...")

    # Get support vector dimensionality
    support_vectors_dim = G.dim_z
    if 'stylegan' in gan:
        if stylegan_space == 'W+':
            support_vectors_dim *= (stylegan_layer + 1)
        elif stylegan_space == 'S':
            support_vectors_dim = sum(list(STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan].values()))

    LSS = LatentSupportSets(num_support_sets=len(semantic_dipoles),
                            num_support_dipoles=args_json.__dict__["num_latent_support_dipoles"],
                            support_vectors_dim=support_vectors_dim,
                            jung_radius=1)

    # Load pre-trained weights and set to evaluation mode
    if args.verbose:
        print("  \\__Pre-trained weights: {}".format(latent_support_sets_model))
    LSS.load_state_dict(torch.load(latent_support_sets_model, map_location=lambda storage, loc: storage))
    if args.verbose:
        print("  \\__Set to evaluation mode")
    LSS.eval()

    # Upload support sets model to GPU
    if use_cuda:
        LSS = LSS.cuda()

    # Set number of generative paths
    num_gen_paths = LSS.num_support_sets

    # Create output dir for generated images
    out_dir = osp.join(args.exp, 'results', args.pool + model_iter,
                       '{}_{}_{}'.format(2 * args.shift_steps, args.eps, round(2 * args.shift_steps * args.eps, 3)))
    os.makedirs(out_dir, exist_ok=True)

    # TODO: add comment
    if learn_gammas:
        gamma_css_json_file = osp.join(args.exp, 'gamma_css.json')
        with open(gamma_css_json_file, 'r') as f:
            gamma_css_dict = json.load(f)
        GAMMA_CSS = np.zeros((len(semantic_dipoles), 2, len(gamma_css_dict)))
        i = 0
        for k, v in gamma_css_dict.items():
            for j in range(len(semantic_dipoles)):
                GAMMA_CSS[j, 0, i] = v[j][0]
                GAMMA_CSS[j, 1, i] = v[j][1]
            i += 1

        # TODO: add comment
        gamma_figs_dir = osp.join(args.exp, 'results', 'vl_gammas')
        os.makedirs(gamma_figs_dir,  exist_ok=True)
        for j in range(len(semantic_dipoles)):
            gammas_j_pos = GAMMA_CSS[j, 0, :]
            gammas_j_neg = GAMMA_CSS[j, 1, :]
            plt.plot(gammas_j_pos, label=semantic_dipoles[j][0])
            plt.plot(gammas_j_neg, label=semantic_dipoles[j][1])
            plt.legend(loc='upper right')
            plt.savefig(fname=osp.join(gamma_figs_dir, 'gammas_dipole_{}.svg'.format(j)), dpi=300, pad_inches=0)
            plt.savefig(fname=osp.join(gamma_figs_dir, 'gammas_dipole_{}.png'.format(j)), dpi=300, pad_inches=0)
            plt.clf()

    # Set default batch size
    if args.batch_size is None:
        args.batch_size = 2 * args.shift_steps + 1

    ## ============================================================================================================== ##
    ##                                                                                                                ##
    ##                                              [Latent Codes Pool]                                               ##
    ##                                                                                                                ##
    ## ============================================================================================================== ##
    # Get latent codes from the given pool
    if args.verbose:
        print("#. Use latent codes from pool {}...".format(args.pool))
    latent_codes_dirs = [dI for dI in os.listdir(pool) if os.path.isdir(os.path.join(pool, dI))]
    latent_codes_dirs.sort()
    latent_codes_list = [torch.load(osp.join(pool, subdir, 'latent_code_{}.pt'.format('w+' if args.w_space else 'z')),
                                    map_location=lambda storage, loc: storage) for subdir in latent_codes_dirs]

    # Get latent codes in torch Tensor format -- xs refers to z or w+ codes
    xs = torch.cat(latent_codes_list)
    if use_cuda:
        xs = xs.cuda()
    num_of_latent_codes = xs.size()[0]

    ## ============================================================================================================== ##
    ##                                                                                                                ##
    ##                                            [Latent space traversal]                                            ##
    ##                                                                                                                ##
    ## ============================================================================================================== ##
    if args.verbose:
        print("#. Traverse latent space...")
        print("  \\__Experiment                  : {}".format(osp.basename(osp.abspath(args.exp))))
        print("  \\__Number of test latent codes : {}".format(num_of_latent_codes))
        print("  \\__Test latent codes shape     : {}".format(xs.shape))
        print("  \\__Shift magnitude             : {}".format(args.eps))
        print("  \\__Shift steps                 : {}".format(2 * args.shift_steps))
        print("  \\__Traversal length            : {}".format(round(2 * args.shift_steps * args.eps, 3)))

    # Iterate over given latent codes
    for i in range(num_of_latent_codes):
        # Get latent code
        x_ = xs[i, :].unsqueeze(0)

        latent_code_hash = latent_codes_dirs[i]
        if args.verbose:
            update_progress("  \\__.Latent code hash: {} [{:03d}/{:03d}] ".format(latent_code_hash,
                                                                                  i+1,
                                                                                  num_of_latent_codes),
                            num_of_latent_codes, i)

        # Create directory for current latent code
        latent_code_dir = osp.join(out_dir, '{}'.format(latent_code_hash))
        os.makedirs(latent_code_dir, exist_ok=True)

        # Create directory for storing path images
        transformed_images_root_dir = osp.join(latent_code_dir, 'paths_images')
        os.makedirs(transformed_images_root_dir, exist_ok=True)
        transformed_images_strips_root_dir = osp.join(latent_code_dir, 'paths_strips')
        os.makedirs(transformed_images_strips_root_dir, exist_ok=True)

        # Keep all latent paths the current latent code (sample)
        paths_latent_codes = []

        # Keep phi coefficients
        phi_coeffs = dict()

        ## ========================================================================================================== ##
        ##                                                                                                            ##
        ##                                             [ Path Traversal ]                                             ##
        ##                                                                                                            ##
        ## ========================================================================================================== ##
        # Iterate over (interpretable) directions
        for dim in range(num_gen_paths):
            if args.verbose:
                print()
                update_progress("      \\__path: {:03d}/{:03d} ".format(dim + 1, num_gen_paths), num_gen_paths, dim + 1)

            # Create shifted latent codes (for the given latent code z) and generate transformed images
            transformed_images = []

            # REVIEW: Save latent code for the original images
            latent_code = x_
            if (not args.w_space) and ('stylegan' in gan):
                if stylegan_space == 'W':
                    latent_code = G.get_w(x_, truncation=truncation)[:, 0, :]
                    current_path_latent_codes = [latent_code]
                    current_path_latent_shifts = [torch.zeros_like(latent_code).cuda() if use_cuda
                                                  else torch.zeros_like(latent_code)]
                elif stylegan_space == 'W+':
                    latent_code = G.get_w(x_, truncation=truncation)
                    current_path_latent_codes = [latent_code]
                    current_path_latent_shifts = [torch.zeros_like(latent_code).cuda() if use_cuda
                                                  else torch.zeros_like(latent_code)]
                elif stylegan_space == 'S':
                    latent_code = G.get_s(G.get_w(x_, truncation=truncation))
                    current_path_latent_codes = [latent_code]
                    shift_0 = {}
                    for k, v in latent_code.items():
                        shift_0.update({k: torch.zeros_like(v).cuda() if use_cuda else torch.zeros_like(v)})
                    current_path_latent_shifts = [shift_0]
            else:
                current_path_latent_codes = [latent_code]
                current_path_latent_shifts = [torch.zeros_like(latent_code).cuda() if use_cuda
                                              else torch.zeros_like(latent_code)]

            ## ====================================================================================================== ##
            ##                                                                                                        ##
            ##                    [ Traverse through current path (positive/negative directions) ]                    ##
            ##                                                                                                        ##
            ## ====================================================================================================== ##

            # +------------------------------------------------------------------------------------------------------+ #
            # |                                        [ Positive Direction ]                                        | #
            # +------------------------------------------------------------------------------------------------------+ #
            latent_code = x_.clone()
            if (not args.w_space) and ('stylegan' in gan):
                if stylegan_space == 'W':
                    latent_code = G.get_w(x_, truncation=truncation)[:, 0, :].clone()
                elif stylegan_space == 'W+':
                    latent_code = G.get_w(x_, truncation=truncation).clone()
                elif stylegan_space == 'S':
                    latent_code = G.get_s(G.get_w(x_, truncation=truncation)).copy()

            cnt = 0
            for _ in range(args.shift_steps):
                cnt += 1

                # Calculate shift vector based on current z
                support_sets_mask = torch.zeros(1, LSS.num_support_sets)
                support_sets_mask[0, dim] = 1.0
                if use_cuda:
                    support_sets_mask.cuda()

                # REVIEW
                if 'stylegan' in gan:
                    # === StyleGAN on Z-space ===
                    if stylegan_space == 'Z':
                        # Calculate shift vector based on the given z code
                        with torch.no_grad():
                            shift = args.eps * LSS(support_sets_mask, latent_code)
                        # REVIEW:
                        current_path_latent_code = latent_code
                        current_path_latent_shift = shift
                        # Calculate shifted latent code
                        latent_code = latent_code + shift
                    # === StyleGAN on W-space ===
                    elif stylegan_space == 'W':
                        # Calculate shift vector based on the given w code
                        with torch.no_grad():
                            shift = args.eps * LSS(support_sets_mask, latent_code)
                        # REVIEW:
                        current_path_latent_code = latent_code
                        current_path_latent_shift = shift
                        # Calculate shifted latent code
                        latent_code = latent_code + shift
                    # === StyleGAN on W+-space ===
                    elif stylegan_space == 'W+':
                        # Calculate shift vector based on the given w+ code, only on the selected layers (i.e., on the
                        # first stylegan_layer + 1 layers)
                        with torch.no_grad():
                            shift = args.eps * LSS(support_sets_mask, latent_code[:, :stylegan_layer + 1, :].reshape(latent_code.shape[0], -1))
                            shift = F.pad(input=shift, pad=(0, (STYLEGAN_LAYERS[gan] - 1 - stylegan_layer) * 512),
                                          mode='constant', value=0).reshape_as(latent_code)
                        # REVIEW:
                        current_path_latent_code = latent_code
                        current_path_latent_shift = shift
                        # Calculate shifted latent code
                        latent_code = latent_code + shift
                    # === StyleGAN on S-space ===
                    elif stylegan_space == 'S':
                        # Calculate shift vector based on the given s code (styles dictionary), only on the target
                        # layers (i.e., those defined by STYLEGAN2_STYLE_SPACE_TARGET_LAYERS for the given GAN)
                        shift_target_styles_vector = args.eps * LSS(support_sets_mask, torch.cat(
                            [latent_code[k] for k in STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan].keys()], dim=1))
                        shift_target_styles_tuple = torch.split(
                            tensor=shift_target_styles_vector,
                            split_size_or_sections=list(STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan].values()),
                            dim=1)
                        latent_code_target_styles_vector = torch.cat(
                            [latent_code[k] for k in STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan].keys()], dim=1)
                        latent_code_target_styles_vector = latent_code_target_styles_vector + shift_target_styles_vector
                        latent_code_target_styles_tuple = torch.split(
                            tensor=latent_code_target_styles_vector,
                            split_size_or_sections=list(STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan].values()),
                            dim=1)
                        shift = dict()
                        latent_code_shifted = dict()
                        style_cnt = 0
                        for k, v in latent_code.items():
                            if k in STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan]:
                                latent_code_shifted.update({k: latent_code_target_styles_tuple[style_cnt]})
                                shift.update({k: shift_target_styles_tuple[style_cnt]})
                                style_cnt += 1
                            else:
                                latent_code_shifted.update({k: latent_code[k]})
                                shift.update({k: torch.zeros_like(latent_code[k])})
                        # REVIEW:
                        current_path_latent_code = latent_code
                        current_path_latent_shift = shift
                        # Calculate shifted latent code
                        latent_code = latent_code_shifted

                else:
                    # Calculate shift vector based on the given z-code
                    with torch.no_grad():
                        shift = args.eps * LSS(support_sets_mask, latent_code)
                    # REVIEW:
                    current_path_latent_code = latent_code
                    current_path_latent_shift = shift
                    # Calculate shifted latent code
                    latent_code = latent_code + shift

                # Store latent codes and shifts
                if cnt == args.shift_leap:
                    current_path_latent_codes.append(current_path_latent_code)
                    current_path_latent_shifts.append(current_path_latent_shift)
                    cnt = 0

            # positive_endpoint = latent_code.clone().reshape(1, -1)
            # ========================

            # +------------------------------------------------------------------------------------------------------+ #
            # |                                        [ Negative Direction ]                                        | #
            # +------------------------------------------------------------------------------------------------------+ #
            latent_code = x_.clone()
            if (not args.w_space) and ('stylegan' in gan):
                if stylegan_space == 'W':
                    latent_code = G.get_w(x_, truncation=truncation)[:, 0, :].clone()
                elif stylegan_space == 'W+':
                    latent_code = G.get_w(x_, truncation=truncation).clone()
                elif stylegan_space == 'S':
                    latent_code = G.get_s(G.get_w(x_, truncation=truncation)).copy()

            cnt = 0
            for _ in range(args.shift_steps):
                cnt += 1

                # Calculate shift vector based on current z
                support_sets_mask = torch.zeros(1, LSS.num_support_sets)
                support_sets_mask[0, dim] = 1.0
                if use_cuda:
                    support_sets_mask.cuda()

                # REVIEW
                if 'stylegan' in gan:
                    # === StyleGAN on Z-space ===
                    if stylegan_space == 'Z':
                        # Calculate shift vector based on the given z code
                        with torch.no_grad():
                            shift = -args.eps * LSS(support_sets_mask, latent_code)
                        # REVIEW:
                        current_path_latent_code = latent_code
                        current_path_latent_shift = shift
                        # Calculate shifted latent code
                        latent_code = latent_code + shift
                    # === StyleGAN on W-space ===
                    elif stylegan_space == 'W':
                        # Calculate shift vector based on the given w code
                        with torch.no_grad():
                            shift = -args.eps * LSS(support_sets_mask, latent_code)
                        # REVIEW:
                        current_path_latent_code = latent_code
                        current_path_latent_shift = shift
                        # Calculate shifted latent code
                        latent_code = latent_code + shift
                    # === StyleGAN on W+-space ===
                    elif stylegan_space == 'W+':
                        # Calculate shift vector based on the given w+ code, only on the selected layers (i.e., on the
                        # first stylegan_layer + 1 layers)
                        with torch.no_grad():
                            shift = -args.eps * LSS(support_sets_mask,
                                                    latent_code[:, :stylegan_layer + 1, :].reshape(latent_code.shape[0],
                                                                                                   -1))
                            shift = F.pad(input=shift, pad=(0, (STYLEGAN_LAYERS[gan] - 1 - stylegan_layer) * 512),
                                          mode='constant', value=0).reshape_as(latent_code)
                        # REVIEW:
                        current_path_latent_code = latent_code
                        current_path_latent_shift = shift
                        # Calculate shifted latent code
                        latent_code = latent_code + shift
                    # === StyleGAN on S-space ===
                    elif stylegan_space == 'S':
                        # Calculate shift vector based on the given s code (styles dictionary), only on the target
                        # layers (i.e., those defined by STYLEGAN2_STYLE_SPACE_TARGET_LAYERS for the given GAN)
                        shift_target_styles_vector = -args.eps * LSS(support_sets_mask, torch.cat(
                            [latent_code[k] for k in STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan].keys()], dim=1))
                        shift_target_styles_tuple = torch.split(
                            tensor=shift_target_styles_vector,
                            split_size_or_sections=list(STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan].values()),
                            dim=1)
                        latent_code_target_styles_vector = torch.cat(
                            [latent_code[k] for k in STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan].keys()], dim=1)
                        latent_code_target_styles_vector = latent_code_target_styles_vector + shift_target_styles_vector
                        latent_code_target_styles_tuple = torch.split(
                            tensor=latent_code_target_styles_vector,
                            split_size_or_sections=list(STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan].values()),
                            dim=1)
                        shift = dict()
                        latent_code_shifted = dict()
                        style_cnt = 0
                        for k, v in latent_code.items():
                            if k in STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[gan]:
                                latent_code_shifted.update({k: latent_code_target_styles_tuple[style_cnt]})
                                shift.update({k: shift_target_styles_tuple[style_cnt]})
                                style_cnt += 1
                            else:
                                latent_code_shifted.update({k: latent_code[k]})
                                shift.update({k: torch.zeros_like(latent_code[k])})
                        # REVIEW:
                        current_path_latent_code = latent_code
                        current_path_latent_shift = shift
                        # Calculate shifted latent code
                        latent_code = latent_code_shifted

                else:
                    # Calculate shift vector based on the given z-code
                    with torch.no_grad():
                        shift = args.eps * LSS(support_sets_mask, latent_code)
                    # REVIEW:
                    current_path_latent_code = latent_code
                    current_path_latent_shift = shift
                    # Calculate shifted latent code
                    latent_code = latent_code + shift

                # Store latent codes and shifts
                if cnt == args.shift_leap:
                    current_path_latent_codes = [current_path_latent_code] + current_path_latent_codes
                    current_path_latent_shifts = [current_path_latent_shift] + current_path_latent_shifts
                    cnt = 0

            # negative_endpoint = latent_code.clone().reshape(1, -1)
            # # ========================

            # Calculate latent path phi coefficient (end-to-end distance / latent path length)
            # phi = torch.norm(negative_endpoint - positive_endpoint, dim=1).item() / (2 * args.shift_steps * args.eps)
            # phi_coeffs.update({dim: phi})

            # Generate transformed images
            if ('stylegan' in gan) and (stylegan_space == 'S'):
                # TODO: add comment
                current_path_latent_codes_list_batches = list()
                for t in range(0, len(current_path_latent_codes), args.batch_size):
                    current_path_latent_codes_list_batches.append(current_path_latent_codes[t:t+args.batch_size])

                current_path_latent_codes_batches = list()
                for item in current_path_latent_codes_list_batches:
                    d = dict()
                    for k in list(latent_code.keys()):
                        d[k] = torch.cat([dd[k] for dd in item], dim=0)
                    current_path_latent_codes_batches.append(d)

                # TODO: add comment
                current_path_latent_shifts_list_batches = list()
                for t in range(0, len(current_path_latent_shifts), args.batch_size):
                    current_path_latent_shifts_list_batches.append(current_path_latent_shifts[t:t + args.batch_size])

                current_path_latent_shifts_batches = list()
                for item in current_path_latent_shifts_list_batches:
                    d = dict()
                    for k in list(latent_code.keys()):
                        d[k] = torch.cat([dd[k] for dd in item], dim=0)
                    current_path_latent_shifts_batches.append(d)

                # TODO: add comment
                transformed_img = []
                for t in range(len(current_path_latent_codes_batches)):
                    # Add dictionaries `current_path_latent_codes_batches[t] + current_path_latent_shifts_batches[t]`
                    d = dict()
                    for k in list(latent_code.keys()):
                        d[k] = current_path_latent_codes_batches[t][k] + current_path_latent_shifts_batches[t][k]
                    with torch.no_grad():
                        transformed_img.append(G(d))
                transformed_img = torch.cat(transformed_img)
                # print()
                # print("transformed_img.shape: {}".format(transformed_img.shape))
                # sys.exit()
            else:
                # Split latent codes and shifts in batches
                current_path_latent_codes = torch.cat(current_path_latent_codes)
                current_path_latent_codes_batches = torch.split(current_path_latent_codes, args.batch_size)
                current_path_latent_shifts = torch.cat(current_path_latent_shifts)
                current_path_latent_shifts_batches = torch.split(current_path_latent_shifts, args.batch_size)
                if len(current_path_latent_codes_batches) != len(current_path_latent_shifts_batches):
                    raise AssertionError()
                else:
                    num_batches = len(current_path_latent_codes_batches)

                transformed_img = []
                for t in range(num_batches):
                    with torch.no_grad():
                        transformed_img.append(G(current_path_latent_codes_batches[t] +
                                                 current_path_latent_shifts_batches[t]))
                transformed_img = torch.cat(transformed_img)

            # Convert tensors (transformed images) into PIL images
            for t in range(transformed_img.shape[0]):
                transformed_images.append(tensor2image(transformed_img[t, :].cpu(),
                                                       img_size=args.img_size,
                                                       adaptive=True))
            # Save all images in `transformed_images` list under `transformed_images_root_dir/<path_<dim>/`
            transformed_images_dir = osp.join(transformed_images_root_dir, 'path_{:03d}'.format(dim))
            os.makedirs(transformed_images_dir, exist_ok=True)

            for t in range(len(transformed_images)):
                transformed_images[t].save(osp.join(transformed_images_dir, '{:06d}.jpg'.format(t)),
                                           "JPEG", quality=args.img_quality, optimize=True, progressive=True)
                # Save original image
                if (t == len(transformed_images) // 2) and (dim == 0):
                    transformed_images[t].save(osp.join(latent_code_dir, 'original_image.jpg'),
                                               "JPEG", quality=95, optimize=True, progressive=True)

            # Create strip of images
            transformed_images_strip = create_strip(image_list=transformed_images, N=args.strip_number,
                                                    strip_height=args.strip_height)
            transformed_images_strip.save(osp.join(transformed_images_strips_root_dir,
                                                   'path_{:03d}_strip.jpg'.format(dim)),
                                          "JPEG", quality=args.img_quality, optimize=True, progressive=True)

            # Save gif (static original image + traversal gif)
            transformed_images_gif_frames = create_gif(transformed_images, gif_height=args.gif_height)
            im = Image.new(mode='RGB', size=(2 * args.gif_height, args.gif_height))
            im.save(fp=osp.join(transformed_images_strips_root_dir, 'path_{:03d}.gif'.format(dim)),
                    append_images=transformed_images_gif_frames,
                    save_all=True,
                    optimize=True,
                    loop=0,
                    duration=1000 // args.gif_fps)

            # Append latent paths
            # paths_latent_codes.append(current_path_latent_codes.unsqueeze(0))

            if args.verbose:
                update_stdout(1)
        # ============================================================================================================ #

        # Save all latent paths and shifts for the current latent code (sample) in a tensor of size:
        #   paths_latent_codes : torch.Size([num_gen_paths, 2 * args.shift_steps + 1, G.dim_z])
        # torch.save(torch.cat(paths_latent_codes), osp.join(latent_code_dir, 'paths_latent_codes.pt'))

        if args.verbose:
            update_stdout(1)
            print()
            print()

    # Create summarizing MD files
    if args.gif or args.strip:
        # For each interpretable path (warping function), collect the generated image sequences for each original latent
        # code and collate them into a GIF file
        print("#. Write summarizing MD files...")

        # Write .md summary files
        if args.gif:
            md_summary_file = osp.join(out_dir, 'results.md')
            md_summary_file_f = open(md_summary_file, "w")
            md_summary_file_f.write("# Experiment: {}\n".format(args.exp))

        if args.strip:
            md_summary_strips_file = osp.join(out_dir, 'results_strips.md')
            md_summary_strips_file_f = open(md_summary_strips_file, "w")
            md_summary_strips_file_f.write("# Experiment: {}\n".format(args.exp))

        if args.gif or args.strip:
            for dim in range(num_gen_paths):
                # Append to .md summary files
                if args.gif:
                    md_summary_file_f.write("### \"{}\" &#8594; \"{}\"\n".format(semantic_dipoles[dim][1],
                                                                                 semantic_dipoles[dim][0]))
                    md_summary_file_f.write("<p align=\"center\">\n")
                if args.strip:
                    md_summary_strips_file_f.write("## \"{}\" &#8594; \"{}\"\n".format(semantic_dipoles[dim][1],
                                                                                       semantic_dipoles[dim][0]))
                    md_summary_strips_file_f.write("<p align=\"center\">\n")

                for lc in latent_codes_dirs:
                    if args.gif:
                        md_summary_file_f.write("<img src=\"{}\" width=\"450\" class=\"center\"/>\n".format(
                            osp.join(lc, 'paths_strips', 'path_{:03d}.gif'.format(dim))))
                    if args.strip:
                        md_summary_strips_file_f.write("<img src=\"{}\" style=\"width: 75vw\"/>\n".format(
                            osp.join(lc, 'paths_strips', 'path_{:03d}_strip.jpg'.format(dim))))
                if args.gif:
                    # md_summary_file_f.write("phi={}\n".format(phi_coeffs[dim]))
                    md_summary_file_f.write("</p>\n")
                if args.strip:
                    # md_summary_strips_file_f.write("phi={}\n".format(phi_coeffs[dim]))
                    md_summary_strips_file_f.write("</p>\n")

        if args.gif:
            md_summary_file_f.close()
        if args.strip:
            md_summary_strips_file_f.close()


if __name__ == '__main__':
    main()
