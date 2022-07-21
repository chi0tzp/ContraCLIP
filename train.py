import sys
import argparse
import os.path as osp
import json
import torch
import clip
from lib import *
from lib import GENFORCE_MODELS, STYLEGAN_LAYERS, SEMANTIC_DIPOLES_CORPORA, STYLEGAN2_STYLE_SPACE_TARGET_LAYERS
from models.load_generator import load_generator


def main():
    """ContraCLIP -- Training script.

    Options:
        ===[ GAN Generator (G) ]========================================================================================
        --gan                        : set pre-trained GAN generator (see GENFORCE_MODELS in lib/config.py)
        --stylegan-space             : set StyleGAN's latent space (Z, W, W+, S) to look for interpretable paths
        --stylegan-layer             : choose up to which StyleGAN's layer to use for learning latent paths
                                       E.g., if --stylegan-layer=11, then interpretable paths will be learnt in a
                                       (12 * 512)-dimensional latent space.
        --truncation                 : set W-space truncation parameter. If set, W-space codes will be truncated

        ===[ Corpus Support Sets (CSS) ]================================================================================
        --corpus                     : choose corpus of prompts (see config.py/PROMPT_CORPUS). The number of elements of
                                       the tuple PROMPT_CORPUS[args.corpus] will define the number of the latent support
                                       sets; i.e., the number of warping functions -- number of the interpretable latent
                                       paths to be optimised
        --learn-css-gammas           : optimise CSS RBF gamma parameters
        --lambda-id                  : ID loss weighting parameter
        ===[ Latent Support Sets (LSS) ]================================================================================
        --num-latent-support-dipoles : set number of support dipoles per support set

        --linear                     : use the vector connecting the poles of the dipole for calculating image-text
                                       similarity
        --min-shift-magnitude        : set minimum latent shift magnitude
        --max-shift-magnitude        : set maximum latent shift magnitude



        ===[ Training ]=================================================================================================
        --max-iter                   : set maximum number of training iterations
        --batch-size                 : set training batch size
        --loss                       : set loss function ('cossim', 'contrastive')
        --temperature                : set contrastive loss temperature
        --lr                         : set learning rate for learning the latent support sets LSS (with Adam optimizer)
        --log-freq                   : set number iterations per log
        --ckp-freq                   : set number iterations per checkpoint model saving

        ===[ CUDA ]=====================================================================================================
        --cuda                           : use CUDA during training (default)
        --no-cuda                        : do NOT use CUDA during training
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="ContraCLIP training script")

    # === Experiment ID ============================================================================================== #
    parser.add_argument('--exp-id', type=str, help="aux experiment ID")

    # === Pre-trained GAN Generator (G) ============================================================================== #
    parser.add_argument('--gan', type=str, choices=GENFORCE_MODELS.keys(), help='GAN generator model')
    parser.add_argument('--stylegan-space', type=str, default='Z', choices=('Z', 'W', 'W+', 'S'),
                        help="StyleGAN's latent space")
    parser.add_argument('--stylegan-layer', type=int, default=11, choices=range(18),
                        help="choose up to which StyleGAN's layer to use for learning latent paths")
    parser.add_argument('--truncation', type=float, help="latent code sampling truncation parameter")

    # === Corpus Support Sets (CSS) ================================================================================== #
    parser.add_argument('--corpus', type=str, required=True, choices=SEMANTIC_DIPOLES_CORPORA.keys(),
                        help="choose corpus of semantic dipoles")
    parser.add_argument('--learn-css-gammas', action='store_true', help="optimise CSS RBF gamma parameters")
    # TODO
    # parser.add_argument('--linear', action='store_true', help="")

    # === Latent Support Sets (LSS) ================================================================================== #
    parser.add_argument('--num-latent-support-dipoles', type=int, help="number of latent support dipoles / support set")
    parser.add_argument('--min-shift-magnitude', type=float, default=0.1, help="minimum latent shift magnitude")
    parser.add_argument('--max-shift-magnitude', type=float, default=0.2, help="maximum latent shift magnitude")
    parser.add_argument('--id', action='store_true', help="impose ID preservation using ArcFace loss")
    parser.add_argument('--lambda-id', type=float, default=100, help="ID loss weighting parameter")

    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=10000, help="maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, required=True, help="training batch size -- this should be less than "
                                                                      "or equal to the size of the given corpus")
    parser.add_argument('--loss', type=str, default='cossim', choices=('cossim', 'contrastive'),
                        help="loss function")
    parser.add_argument('--temperature', type=float, default=1.0, help="contrastive temperature")
    parser.add_argument('--lr', type=float, default=1e-3, help="latent support sets (LSS) learning rate")
    parser.add_argument('--log-freq', default=10, type=int, help='number of iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='number of iterations per checkpoint model saving')

    # === CUDA ======================================================================================================= #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Check given batch size
    if args.batch_size > len(SEMANTIC_DIPOLES_CORPORA[args.corpus]):
        print("*** WARNING ***: Given batch size ({}) is greater than the size of the given corpus ({})\n"
              "                 Set batch size to {}".format(
                args.batch_size, len(SEMANTIC_DIPOLES_CORPORA[args.corpus]),
                len(SEMANTIC_DIPOLES_CORPORA[args.corpus])))
        args.batch_size = len(SEMANTIC_DIPOLES_CORPORA[args.corpus])

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
                       latent_is_s=('stylegan' in args.gan) and (args.stylegan_space == 'S'),
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
    prompt_f = PromptFeatures(prompt_corpus=SEMANTIC_DIPOLES_CORPORA[args.corpus],
                              clip_model=clip_model,
                              use_cuda=use_cuda)
    prompt_features = prompt_f.get_prompt_features()

    # REVIEW: Get CSS dipole betas
    # if 'stylegan' in args.gan:
    #     dipole_betas_file = osp.join('experiments', 'css_betas',
    #                                  '{}-W-truncation-{}_{}_betas.json'.format(args.gan, args.truncation, args.corpus))
    # else:
    #     dipole_betas_file = osp.join('experiments', 'css_betas', '{}_{}_betas.json'.format(args.gan, args.corpus))
    # with open(dipole_betas_file, 'r') as f:
    #     dipole_betas_ = json.load(f)
    # dipole_betas = []
    # for i in range(len(SEMANTIC_DIPOLES_CORPORA[args.corpus])):
    #     dipole_betas.append([dipole_betas_[i][SEMANTIC_DIPOLES_CORPORA[args.corpus][i][0]],
    #                          dipole_betas_[i][SEMANTIC_DIPOLES_CORPORA[args.corpus][i][1]]])
    dipole_betas = []
    for i in range(len(SEMANTIC_DIPOLES_CORPORA[args.corpus])):
        dipole_betas.append([0.5, 0.5])

    # Build Corpus Support Sets model CSS
    print("#. Build Corpus Support Sets CSS...")
    print("  \\__Number of corpus support sets    : {}".format(prompt_f.num_prompts))
    print("  \\__Number of corpus support dipoles : {}".format(1))
    print("  \\__Prompt features dim              : {}".format(prompt_f.prompt_features_dim))
    print("  \\__Learn RBF gammas                 : {}".format(args.learn_css_gammas))

    CSS = CorpusSupportSets(prompt_features=prompt_features,
                            dipole_betas=dipole_betas,
                            learn_gammas=args.learn_css_gammas)

    # Count number of trainable parameters
    CSS_trainable_parameters = sum(p.numel() for p in CSS.parameters() if p.requires_grad)
    print("  \\__Trainable parameters             : {:,}".format(CSS_trainable_parameters))

    # Set support vector dimensionality and initial gamma param
    support_vectors_dim = G.dim_z
    if 'stylegan' in args.gan:
        if args.stylegan_space == 'W+':
            support_vectors_dim *= (args.stylegan_layer + 1)
        elif args.stylegan_space == 'S':
            support_vectors_dim = sum(list(STYLEGAN2_STYLE_SPACE_TARGET_LAYERS[args.gan].values()))

    # Get Jung radii
    with open(osp.join('models', 'jung_radii.json'), 'r') as f:
        jung_radii_dict = json.load(f)

    if 'stylegan' in args.gan:
        if args.stylegan_space == 'W+':
            lm = jung_radii_dict[args.gan]['W']['{}'.format(args.stylegan_layer)]
        elif args.stylegan_space == 'W':
            lm = jung_radii_dict[args.gan]['W']['0']
        elif args.stylegan_space == 'S':
            lm = jung_radii_dict[args.gan]['S']
        else:
            lm = jung_radii_dict[args.gan]['Z']
        jung_radius = lm[0] * args.truncation + lm[1]
    else:
        jung_radius = jung_radii_dict[args.gan]['Z'][1]

    # Build Latent Support Sets model LSS
    print("#. Build Latent Support Sets LSS...")
    print("  \\__Number of latent support sets    : {}".format(prompt_f.num_prompts))
    print("  \\__Number of latent support dipoles : {}".format(args.num_latent_support_dipoles))
    print("  \\__Support Vectors dim              : {}".format(support_vectors_dim))
    print("  \\__Jung radius                      : {:.2f}".format(jung_radius))

    LSS = LatentSupportSets(num_support_sets=prompt_f.num_prompts,
                            num_support_dipoles=args.num_latent_support_dipoles,
                            support_vectors_dim=support_vectors_dim,
                            jung_radius=jung_radius)

    # Count number of trainable parameters
    LSS_trainable_parameters = sum(p.numel() for p in LSS.parameters() if p.requires_grad)
    print("  \\__Trainable parameters             : {:,}".format(LSS_trainable_parameters))

    # Build ID loss (ArcFace)
    if args.id:
        print("#. Build ArcFace (ID loss)...")
    id_loss = IDLoss()

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    t = Trainer(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu)

    # Train
    t.train(generator=G, latent_support_sets=LSS, corpus_support_sets=CSS, clip_model=clip_model,
            id_loss=id_loss if args.id else None)


if __name__ == '__main__':
    main()
