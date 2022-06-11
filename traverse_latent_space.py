import argparse
import os
import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import json
from torchvision.transforms import ToPILImage
from lib import SupportSets, GENFORCE_MODELS, update_progress, update_stdout
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


# Check whether this can be replaced by `tensor2image` in lib/aux.py
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


def create_strip(image_list, N=5, size=256):
    step = len(image_list) // N
    transformed_images_strip = Image.new('RGB', (N * size, size))
    for i in range(N):
        # Draw rectangle around the central image
        if i == N // 2:
            draw_rect = ImageDraw.Draw(image_list[i * step].resize(size=(size, size)))
            rect_width = 5
            rect_colour = (175, 5, 25)
            draw_rect.rectangle(xy=((0, 0), (size, size)), outline=rect_colour, width=rect_width)
        transformed_images_strip.paste(image_list[i * step].resize(size=(size, size)), (i * size, 0))

    transformed_images_gif_frames = []
    for i in range(len(image_list)):
        gif_frame = Image.new('RGB', (2 * size, size))
        gif_frame.paste(image_list[len(image_list) // 2].resize(size=(size, size)), (0, 0))
        gif_frame.paste(image_list[i].resize(size=(size, size)), (size, 0))

        # Draw progress bar
        draw_bar = ImageDraw.Draw(gif_frame)
        bar_h = 12
        bar_colour = (252, 186, 3)
        draw_bar.rectangle(xy=((size, size - bar_h), ((1 + (i / len(image_list))) * size, size)), fill=bar_colour)

        # Draw rectangle around the moving part
        # draw_rect = ImageDraw.Draw(gif_frame)
        # rect_width = 5
        # rect_colour = (0, 0, 0)
        # draw_rect.rectangle(xy=((size, 0), (2 * size, size)), outline=rect_colour, width=rect_width)

        transformed_images_gif_frames.append(gif_frame)

    return transformed_images_strip, transformed_images_gif_frames


def get_concat_h(img_file_orig,
                 shifted_img_file,
                 size,
                 s,
                 shift_steps,
                 draw_progress_bar=True):
    img_orig = Image.open(img_file_orig).resize((size, size))
    img_orig_w = img_orig.width
    img_orig_h = img_orig.height

    img_shifted = Image.open(shifted_img_file).resize((size, size))
    img_shifted_w = img_shifted.width

    dst = Image.new('RGB', (img_orig_w + img_shifted_w, img_orig_h))
    dst.paste(img_orig, (0, 0))
    dst.paste(img_shifted, (img_orig_w, 0))

    # Draw progress bar
    if draw_progress_bar:
        draw = ImageDraw.Draw(dst)
        bar_h = 7
        bar_color = (252, 186, 3)
        # draw.rectangle(xy=[(size, size - bar_h), ((1 + s / shift_steps) * size, size)], fill=bar_color)
        draw.rectangle(xy=((size, size - bar_h), ((1 + s / shift_steps) * size, size)), fill=bar_color)

    return dst


def main():
    """ContraCLIP -- Latent space traversal script.

    A script for traversing the latent space of a pre-trained GAN generator through paths defined by the warpings of
    a set of pre-trained support vectors. Latent codes are drawn from a pre-defined collection via the `--pool`
    argument. The generated images are stored under `results/` directory.

    Options:
        ================================================================================================================
        -v, --verbose : set verbose mode on
        ================================================================================================================
        --exp         : set experiment's model dir, as created by `train.py`, i.e., it should contain a sub-directory
                        `models/` with two files, namely `reconstructor.pt` and `support_sets.pt`, which
                        contain the weights for the reconstructor and the support sets, respectively, and an `args.json`
                        file that contains the arguments the model has been trained with.
        --pool        : directory of pre-defined pool of latent codes (created by `sample_gan.py`)
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
        --gif-size    : set GIF image size
        --gif-fps     : set number of frames per second for the generated GIF images
        ================================================================================================================
        --cuda        : use CUDA (default)
        --no-cuda     : do not use CUDA
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="ContraCLIP latent space traversal script")
    parser.add_argument('-v', '--verbose', action='store_true', help="set verbose mode on")
    # ================================================================================================================ #
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
    parser.add_argument('--img-quality', type=int, default=75, help="set JPEG image quality")

    parser.add_argument('--strip-number', type=int, default=5, help="set number of images per strip")

    parser.add_argument('--gif', action='store_true', help="Create GIF traversals")
    # parser.add_argument('--gif-size', type=int, default=256, help="set gif resolution")
    parser.add_argument('--gif-size', type=int, default=196, help="set gif resolution")
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

    # TODO: add comment
    support_vectors_dim = G.dim_z
    if ('stylegan' in gan) and (stylegan_space == 'W+'):
        support_vectors_dim *= (stylegan_layer + 1)

    LSS = SupportSets(num_support_sets=len(semantic_dipoles),
                      num_support_dipoles=args_json.__dict__["num_latent_support_dipoles"],
                      support_vectors_dim=support_vectors_dim)

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
    latent_codes_z = []
    for subdir in latent_codes_dirs:
        latent_codes_z.append(torch.load(osp.join(pool, subdir, 'latent_code_z.pt'),
                                         map_location=lambda storage, loc: storage))
    zs = torch.cat(latent_codes_z)
    if use_cuda:
        zs = zs.cuda()
    num_of_latent_codes = zs.size()[0]

    ## ============================================================================================================== ##
    ##                                                                                                                ##
    ##                                            [Latent space traversal]                                            ##
    ##                                                                                                                ##
    ## ============================================================================================================== ##
    if args.verbose:
        print("#. Traverse latent space...")
        print("  \\__Experiment       : {}".format(osp.basename(osp.abspath(args.exp))))
        print("  \\__Shift magnitude  : {}".format(args.eps))
        print("  \\__Shift steps      : {}".format(2 * args.shift_steps))
        print("  \\__Traversal length : {}".format(round(2 * args.shift_steps * args.eps, 3)))
        print("  \\__Save results at  : {}".format(out_dir))

    # Iterate over given latent codes
    for i in range(num_of_latent_codes):
        # Un-squeeze current latent code in shape [1, dim] and create hash code for it
        z_ = zs[i, :].unsqueeze(0)

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

            # Current path's latent codes and shifts lists
            latent_code = z_
            if ('stylegan' in gan) and ('W' in stylegan_space):
                if stylegan_space == 'W':
                    latent_code = G.get_w(zs[i, :].unsqueeze(0), truncation=truncation)[:, 0, :]
                elif stylegan_space == 'W+':
                    latent_code = G.get_w(zs[i, :].unsqueeze(0), truncation=truncation)
            current_path_latent_codes = [latent_code]
            current_path_latent_shifts = [torch.zeros_like(latent_code).cuda() if use_cuda
                                          else torch.zeros_like(latent_code)]

            ## ====================================================================================================== ##
            ##                                                                                                        ##
            ##                    [ Traverse through current path (positive/negative directions) ]                    ##
            ##                                                                                                        ##
            ## ====================================================================================================== ##
            # == Positive direction ==
            if ('stylegan' in gan) and ('W' in stylegan_space):
                if stylegan_space == 'W':
                    latent_code = G.get_w(z_, truncation=truncation)[:, 0, :].clone()
                elif stylegan_space == 'W+':
                    latent_code = G.get_w(z_, truncation=truncation).clone()
            else:
                latent_code = z_.clone()
            cnt = 0
            for _ in range(args.shift_steps):
                cnt += 1

                # Calculate shift vector based on current z
                support_sets_mask = torch.zeros(1, LSS.num_support_sets)
                support_sets_mask[0, dim] = 1.0
                if use_cuda:
                    support_sets_mask.cuda()

                # Get latent space shift vector and shifted latent code
                if ('stylegan' in gan) and (stylegan_space == 'W+'):
                    with torch.no_grad():
                        shift = args.eps * LSS(support_sets_mask,
                                               latent_code[:, :stylegan_layer + 1, :].reshape(latent_code.shape[0], -1))
                    latent_code = latent_code + \
                        F.pad(input=shift, pad=(0, (17 - stylegan_layer) * 512),
                              mode='constant', value=0).reshape_as(latent_code)
                    current_path_latent_code = latent_code
                else:
                    with torch.no_grad():
                        shift = args.eps * LSS(support_sets_mask, latent_code)
                    latent_code = latent_code + shift
                    current_path_latent_code = latent_code

                # Store latent codes and shifts
                if cnt == args.shift_leap:
                    if ('stylegan' in gan) and (stylegan_space == 'W+'):
                        current_path_latent_shifts.append(F.pad(input=shift, pad=(0, (17 - stylegan_layer) * 512), mode='constant', value=0).reshape_as(latent_code))
                    else:
                        current_path_latent_shifts.append(shift)
                    current_path_latent_codes.append(current_path_latent_code)
                    cnt = 0

            # == Negative direction ==
            if ('stylegan' in gan) and ('W' in stylegan_space):
                if stylegan_space == 'W':
                    latent_code = G.get_w(z_, truncation=truncation)[:, 0, :].clone()
                elif stylegan_space == 'W+':
                    latent_code = G.get_w(z_, truncation=truncation).clone()
            else:
                latent_code = z_.clone()
            cnt = 0
            for _ in range(args.shift_steps):
                cnt += 1
                # Calculate shift vector based on current z
                support_sets_mask = torch.zeros(1, LSS.num_support_sets)
                support_sets_mask[0, dim] = 1.0
                if use_cuda:
                    support_sets_mask.cuda()

                # Get latent space shift vector and shifted latent code
                if ('stylegan' in gan) and (stylegan_space == 'W+'):
                    with torch.no_grad():
                        shift = -args.eps * LSS(support_sets_mask,
                                                latent_code[:, :stylegan_layer + 1, :].reshape(latent_code.shape[0], -1))
                    latent_code = latent_code + \
                        F.pad(input=shift, pad=(0, (17 - stylegan_layer) * 512),
                              mode='constant', value=0).reshape_as(latent_code)
                    current_path_latent_code = latent_code
                else:
                    with torch.no_grad():
                        shift = -args.eps * LSS(support_sets_mask, latent_code)
                    latent_code = latent_code + shift
                    current_path_latent_code = latent_code

                # Store latent codes and shifts
                if cnt == args.shift_leap:
                    if ('stylegan' in gan) and (stylegan_space == 'W+'):
                        current_path_latent_shifts = \
                            [F.pad(input=shift, pad=(0, (17 - stylegan_layer) * 512),
                                   mode='constant', value=0).reshape_as(latent_code)] + current_path_latent_shifts
                    else:
                        current_path_latent_shifts = [shift] + current_path_latent_shifts
                    current_path_latent_codes = [current_path_latent_code] + current_path_latent_codes
                    cnt = 0
            # ========================

            # Generate transformed images
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
                    # transformed_img.append(G(current_path_latent_codes_batches[t]))
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

            # Create strip of images and save in jpg and gif
            transformed_images_strip, transformed_images_gif_frames = create_strip(image_list=transformed_images,
                                                                                   N=args.strip_number,
                                                                                   size=args.gif_size)
            transformed_images_strip.save(osp.join(transformed_images_strips_root_dir,
                                                   'path_{:03d}_strip.jpg'.format(dim)),
                                          "JPEG", quality=args.img_quality, optimize=True, progressive=True)

            # Save strip gif
            # im = Image.new(mode='RGB', size=((args.strip_number + 1) * args.gif_size, args.gif_size))
            im = Image.new(mode='RGB', size=(2 * args.gif_size, args.gif_size))
            im.save(fp=osp.join(transformed_images_strips_root_dir, 'path_{:03d}.gif'.format(dim)),
                    append_images=transformed_images_gif_frames,
                    save_all=True,
                    optimize=True,
                    loop=0,
                    # duration=1000 // args.gif_fps
                    duration=1000 // args.gif_fps
                    )

            # Append latent paths
            paths_latent_codes.append(current_path_latent_codes.unsqueeze(0))

            if args.verbose:
                update_stdout(1)
        # ============================================================================================================ #

        # Save all latent paths and shifts for the current latent code (sample) in a tensor of size:
        #   paths_latent_codes : torch.Size([num_gen_paths, 2 * args.shift_steps + 1, G.dim_z])
        torch.save(torch.cat(paths_latent_codes), osp.join(latent_code_dir, 'paths_latent_codes.pt'))

        if args.verbose:
            update_stdout(1)
            print()
            print()

    # Collate traversal GIFs
    if args.gif:
        # Build results file structure
        structure = dict()
        generated_img_subdirs = [dI for dI in os.listdir(out_dir) if os.path.isdir(osp.join(out_dir, dI)) and
                                 dI != 'paths_gifs']
        generated_img_subdirs.sort()
        for img_id in generated_img_subdirs:
            structure.update({img_id: {}})
            path_images_dir = osp.join(out_dir, '{}'.format(img_id), 'paths_images')
            path_images_subdirs = [dI for dI in os.listdir(path_images_dir)
                                   if os.path.isdir(os.path.join(path_images_dir, dI))]
            path_images_subdirs.sort()
            for item in path_images_subdirs:
                structure[img_id].update({item: [dI for dI in os.listdir(osp.join(path_images_dir, item))
                                                 if osp.isfile(os.path.join(path_images_dir, item, dI))]})

        # Create directory for storing traversal GIFs
        os.makedirs(osp.join(out_dir, 'paths_gifs'), exist_ok=True)

        # For each interpretable path (warping function), collect the generated image sequences for each original latent
        # code and collate them into a GIF file
        print("#. Collate GIFs...")

        # Write .md summary files
        md_summary_file = osp.join(out_dir, 'results.md')
        md_summary_strips_file = osp.join(out_dir, 'results_strips.md')
        md_summary_file_f = open(md_summary_file, "w")
        md_summary_strips_file_f = open(md_summary_strips_file, "w")
        md_summary_file_f.write("# Experiment: {}\n".format(args.exp))
        md_summary_strips_file_f.write("# Experiment: {}\n".format(args.exp))

        num_of_frames = list()
        for dim in range(num_gen_paths):
            if args.verbose:
                update_progress("  \\__path: {:03d}/{:03d} ".format(dim + 1, num_gen_paths), num_gen_paths, dim + 1)

            gif_frames = []
            for img_id in structure.keys():
                original_img_file = osp.join(out_dir, '{}'.format(img_id), 'original_image.jpg')
                shifted_images_dir = osp.join(out_dir, '{}'.format(img_id), 'paths_images', 'path_{:03d}'.format(dim))

                row_frames = []
                img_id_num_of_frames = 0
                for t in range(len(structure[img_id]['path_{:03d}'.format(dim)])):
                    img_id_num_of_frames += 1
                for t in range(len(structure[img_id]['path_{:03d}'.format(dim)])):
                    shifted_img_file = osp.join(shifted_images_dir, '{:06d}.jpg'.format(t))
                    # Concatenate `original_img_file` and `shifted_img_file`
                    row_frames.append(get_concat_h(img_file_orig=original_img_file,
                                                   shifted_img_file=shifted_img_file,
                                                   size=args.gif_size,
                                                   s=t,
                                                   shift_steps=img_id_num_of_frames))
                num_of_frames.append(img_id_num_of_frames)
                gif_frames.append(row_frames)

            if len(set(num_of_frames)) > 1:
                print("#. Warning: Inconsistent number of frames for image sequences: {}".format(num_of_frames))

            # Create full GIF frames
            full_gif_frames = []
            for f in range(int(num_of_frames[0])):
                gif_f = Image.new('RGB', (2 * args.gif_size, len(structure) * args.gif_size))
                for i in range(len(structure)):
                    gif_f.paste(gif_frames[i][f], (0, i * args.gif_size))
                full_gif_frames.append(gif_f)

            # Save gif
            im = Image.new(mode='RGB', size=(2 * args.gif_size, len(structure) * args.gif_size))
            gif_file = osp.join(out_dir, 'paths_gifs', 'path_{:03d}.gif'.format(dim))
            im.save(
                fp=gif_file,
                append_images=full_gif_frames,
                save_all=True,
                optimize=True,
                loop=0,
                duration=1000 // args.gif_fps)

            # Append to .md summary files
            md_summary_file_f.write("## \"{}\" &#8594; \"{}\"\n".format(semantic_dipoles[dim][1],
                                                                        semantic_dipoles[dim][0]))
            md_summary_file_f.write("<img src=\"{}\" width=\"450\"/>\n".format(
                osp.join('paths_gifs', 'path_{:03d}.gif'.format(dim))))

            md_summary_strips_file_f.write("## \"{}\" &#8594; \"{}\"\n".format(semantic_dipoles[dim][1],
                                                                               semantic_dipoles[dim][0]))
            for i in range(num_of_latent_codes):
                md_summary_strips_file_f.write("<img src=\"{}\" style=\"width: 75vw\"/>\n".format(
                    osp.join('{}'.format(latent_codes_dirs[i]), 'paths_strips', 'path_{:03d}.gif'.format(dim))))

        md_summary_file_f.close()
        md_summary_strips_file_f.close()


if __name__ == '__main__':
    main()