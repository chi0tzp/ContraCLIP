import argparse
import os.path as osp
import json
import torch
import clip
from lib import *
from lib import GENFORCE_MODELS, STYLEGAN_LAYERS, SEMANTIC_DIPOLES_CORPORA
from models.load_generator import load_generator


def main():
    """ContraCLIP -- Training script.

    Options:
        ===[ GAN Generator (G) ]========================================================================================
        --gan                        : set pre-trained GAN generator (see GENFORCE_MODELS in lib/config.py)
        --stylegan-space             : set StyleGAN's latent space (Z, W, W+) to look for interpretable paths
        --stylegan-layer             : choose up to which StyleGAN's layer to use for learning latent paths
                                       E.g., if --stylegan-layer=11, then interpretable paths will be learnt in a
                                       (12 * 512)-dimensional latent space.
        --truncation                 : set W-space truncation parameter. If set, W-space codes will be truncated

        ===[ Corpus Support Sets (CSS) ]================================================================================
        --corpus                     : choose corpus of prompts (see config.py/PROMPT_CORPUS). The number of elements of
                                       the tuple PROMPT_CORPUS[args.corpus] will define the number of the latent support
                                       sets; i.e., the number of warping functions -- number of the interpretable latent
                                       paths to be optimised
        --css-beta                   : set beta parameter for fixing CLIP space RBFs' gamma parameters
                                       (0.25 <= css_beta < 1.0)
        --styleclip                  : use StyleCLIP approach for calculating image-text similarity

        ===[ Latent Support Sets (LSS) ]================================================================================
        --num-latent-support-dipoles : set number of support dipoles per support set
        --lss-beta                   : set beta parameter for initializing latent space RBFs' gamma parameters
                                       (0.0 < lss_beta < 1.0)
        --lr                         : set learning rate for learning the latent support sets LSS (with Adam optimizer)
        --linear                     : use the vector connecting the poles of the dipole for calculating image-text
                                       similarity
        --min-shift-magnitude        : set minimum latent shift magnitude
        --max-shift-magnitude        : set maximum latent shift magnitude

        ===[ CLIP ]=====================================================================================================


        ===[ Training ]=================================================================================================
        --max-iter                   : set maximum number of training iterations
        --batch-size                 : set training batch size
        --loss                       : set loss function ('cossim', 'contrastive')
        --temperature                : set contrastive loss temperature
        --log-freq                   : set number iterations per log
        --ckp-freq                   : set number iterations per checkpoint model saving

        ===[ CUDA ]=====================================================================================================
        --cuda                           : use CUDA during training (default)
        --no-cuda                        : do NOT use CUDA during training
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="WarpedGANSpace training script")

    # === Experiment ID ============================================================================================== #
    parser.add_argument('--exp-id', type=str, default='', help="set optional experiment ID")

    # === Pre-trained GAN Generator (G) ============================================================================== #
    parser.add_argument('--gan', type=str, choices=GENFORCE_MODELS.keys(), help='GAN generator model')
    parser.add_argument('--stylegan-space', type=str, default='Z', choices=('Z', 'W', 'W+'),
                        help="StyleGAN's latent space")
    parser.add_argument('--stylegan-layer', type=int, default=11, choices=range(18),
                        help="choose up to which StyleGAN's layer to use for learning latent paths")
    parser.add_argument('--truncation', type=float, help="latent code sampling truncation parameter")

    # === Corpus Support Sets (CSS) ================================================================================== #
    parser.add_argument('--corpus', type=str, required=True, choices=SEMANTIC_DIPOLES_CORPORA.keys(),
                        help="choose corpus of semantic dipoles")
    parser.add_argument('--css-beta', type=float, default=0.5,
                        help="set beta parameter for initializing CLIP space RBFs' gamma parameters "
                             "(0.25 <= css_beta < 1.0)")
    parser.add_argument('--styleclip', action='store_true',
                        help="use StyleCLIP approach for calculating image-text similarity")
    parser.add_argument('--linear', action='store_true',
                        help="use the vector connecting the poles of the dipole for calculating image-text similarity")

    # === Latent Support Sets (LSS) ================================================================================== #
    parser.add_argument('--num-latent-support-dipoles', type=int, help="number of latent support dipoles / support set")
    parser.add_argument('--lss-beta', type=float, default=0.1,
                        help="set beta parameter for initializing latent space RBFs' gamma parameters "
                             "(0.0 < css_beta < 1.0)")
    parser.add_argument('--lr', type=float, default=1e-3, help="latent support sets LSS learning rate")
    parser.add_argument('--min-shift-magnitude', type=float, default=0.25, help="minimum latent shift magnitude")
    parser.add_argument('--max-shift-magnitude', type=float, default=0.45, help="maximum latent shift magnitude")

    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=10000, help="maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=32, help="training batch size")
    parser.add_argument('--loss', type=str, default='cossim', choices=('cossim', 'contrastive'),
                        help="loss function")
    parser.add_argument('--temperature', type=float, default=1.0, help="contrastive temperature")
    parser.add_argument('--log-freq', default=10, type=int, help='number of iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='number of iterations per checkpoint model saving')

    # === CUDA ======================================================================================================= #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Check StyleGAN's layer
    if 'stylegan' in args.gan:
        if (args.stylegan_layer < 0) or (args.stylegan_layer > STYLEGAN_LAYERS[args.gan]-1):
            raise ValueError("Invalid stylegan_layer for given GAN ({}). Choose between 0 and {}".format(
                args.gan, STYLEGAN_LAYERS[args.gan]-1))

    # Create output dir and save current arguments
    exp_dir = create_exp_dir(args)

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
    print("#. Build GAN generator model G and load with pre-trained weights...")
    print("  \\__GAN generator : {} (res: {})".format(args.gan, GENFORCE_MODELS[args.gan][1]))
    print("  \\__Pre-trained weights: {}".format(GENFORCE_MODELS[args.gan][0]))
    G = load_generator(model_name=args.gan,
                       latent_is_w=('stylegan' in args.gan) and ('W' in args.stylegan_space),
                       verbose=True).eval()

    # Upload GAN generator model to GPU
    if use_cuda:
        G = G.cuda()

    # Build pretrained CLIP model
    print("#. Build pretrained CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device='cuda' if use_cuda else 'cpu', jit=False)
    clip_model.float()
    clip_model.eval()

    # Get CLIP (non-normalized) text features for the prompts of the given corpus
    prompt_f = PromptFeatures(prompt_corpus=SEMANTIC_DIPOLES_CORPORA[args.corpus], clip_model=clip_model)
    prompt_features = prompt_f.get_prompt_features()

    # Build Corpus Support Sets model CSS
    print("#. Build Corpus Support Sets CSS...")
    print("  \\__Number of corpus support sets    : {}".format(prompt_f.num_prompts))
    print("  \\__Number of corpus support dipoles : {}".format(1))
    print("  \\__Prompt features dim              : {}".format(prompt_f.prompt_features_dim))
    print("  \\__Text RBF beta param              : {}".format(args.css_beta))

    CSS = SupportSets(prompt_features=prompt_features, css_beta=args.css_beta)

    # Count number of trainable parameters
    CSS_trainable_parameters = sum(p.numel() for p in CSS.parameters() if p.requires_grad)
    print("  \\__Trainable parameters: {:,}".format(CSS_trainable_parameters))

    # Set support vector dimensionality and initial gamma param
    support_vectors_dim = G.dim_z
    if ('stylegan' in args.gan) and (args.stylegan_space == 'W+'):
        support_vectors_dim *= (args.stylegan_layer + 1)

    # Get expected latent norm
    with open(osp.join('models', 'expected_latent_norms.json'), 'r') as f:
        expected_latent_norms_dict = json.load(f)

    if 'stylegan' in args.gan:
        if 'W+' in args.stylegan_space:
            lm = expected_latent_norms_dict[args.gan]['W']['{}'.format(args.stylegan_layer)]
        elif 'W' in args.stylegan_space:
            lm = expected_latent_norms_dict[args.gan]['W']['0']
        else:
            lm = expected_latent_norms_dict[args.gan]['Z']
        expected_latent_norm = lm[0] * args.truncation + lm[1]
    else:
        expected_latent_norm = expected_latent_norms_dict[args.gan]['Z'][1]

    # Build Latent Support Sets model LSS
    print("#. Build Latent Support Sets LSS...")
    print("  \\__Number of latent support sets    : {}".format(prompt_f.num_prompts))
    print("  \\__Number of latent support dipoles : {}".format(args.num_latent_support_dipoles))
    print("  \\__Support Vectors dim              : {}".format(support_vectors_dim))
    print("  \\__Latent RBF beta param            : {}".format(args.lss_beta))

    LSS = SupportSets(num_support_sets=prompt_f.num_prompts,
                      num_support_dipoles=args.num_latent_support_dipoles,
                      support_vectors_dim=support_vectors_dim,
                      lss_beta=args.lss_beta,
                      expected_latent_norm=expected_latent_norm)

    # Count number of trainable parameters
    LSS_trainable_parameters = sum(p.numel() for p in LSS.parameters() if p.requires_grad)
    print("  \\__Trainable parameters: {:,}".format(LSS_trainable_parameters))
    
    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    t = Trainer(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu)

    # Train
    t.train(generator=G, latent_support_sets=LSS, corpus_support_sets=CSS, clip_model=clip_model)


if __name__ == '__main__':
    main()
