import sys
import argparse
import os.path as osp
import torch
import clip
from lib import *
from models.load_generator import load_generator


def main():
    """ContraCLIP -- Training script. You may find auxiliary training scripts under scripts/train/.

    Options:
        === [Experiment ID] ============================================================================================
        --exp-id                     : optional experiment id string
        --exp-subdir                 : optional subdirectory under experiments/complete/ where completed experiment will
                                       be transferred

        ===[ Latent Paths Network (LPN) ]===============================================================================
        --latent-paths-type          : TODO: (mlp, wgs)






        --num-latent-support-dipoles : set number of support dipoles per support set in the GAN's latent space

        --min-shift-magnitude        : set minimum latent shift magnitude
        --max-shift-magnitude        : set maximum latent shift magnitude


        ===[ GAN Generator (G) ]========================================================================================
        --gan                        : set pre-trained GAN generator (see GENFORCE_MODELS in lib/config.py)
        --stylegan-space             : set StyleGAN's latent space (Z, W, W+, S) to look for interpretable paths
        --stylegan-layer             : choose up to which StyleGAN's W+ layer to use for learning latent paths
                                       E.g., if --stylegan-layer=11, then interpretable paths will be learnt in a
                                       (12 * 512)-dimensional latent space (i.e., the first 12 layers of the W+ space)
        --truncation                 : set GAN's sampling truncation parameter






        ===[ Corpus Support Sets (CSS) ]================================================================================
        --vl-model                   : choose VL model ('clip' or 'farl')
        --corpus                     : choose corpus of semantic dipoles (i.e., a set of pairs of contrasting sentences
                                       in natural language) by giving a key of the dictionary SEMANTIC_DIPOLES_CORPORA
                                       found in lib/config.py. You may define new corpora of semantic dipoles following
                                       the given format
        --gamma-0                    : set initial gamma parameter
        TODO:
        --vl-sim                     : type of VL similarity ('standard', 'proposed-no-warping', 'proposed-warping')
        --include-cls-in-mean        : include the CLS token into calculating text features statistics



        ===[ Training ]=================================================================================================
        --max-iter                   : set maximum number of training iterations
        --batch-size                 : set training batch size
        --temperature                : set contrastive loss temperature
        --lr                         : set learning rate for learning the latent support sets LSS (with Adam optimizer)
        --id                         : impose ID preservation using ArcFace loss
        --lambda-id                  : ID loss weighting parameter
        --log-freq                   : set number iterations per log
        --ckp-freq                   : set number iterations per checkpoint model saving

        ===[ CUDA ]=====================================================================================================
        --cuda                       : use CUDA during training (default)
        --no-cuda                    : do NOT use CUDA during training
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="ContraCLIP training script")

    # === Experiment ID ============================================================================================== #
    parser.add_argument('--exp-id', type=str, help="aux experiment ID")
    parser.add_argument('--exp-subdir', type=str, help="optional subdirectory under experiments/complete/")

    # === Pre-trained GAN Generator (G) ============================================================================== #
    parser.add_argument('--gan', type=str, choices=GENFORCE_MODELS.keys(), help='GAN generator model')
    parser.add_argument('--stylegan-space', type=str, default='Z', choices=('Z', 'W', 'W+', 'S'),
                        help="StyleGAN's latent space")
    parser.add_argument('--stylegan-layer', type=int, default=11, choices=range(18),
                        help="choose up to which StyleGAN's layer to use for learning latent paths")
    parser.add_argument('')





    parser.add_argument('--truncation', type=float, help="latent code sampling truncation parameter")

    # === Corpus Support Sets (CSS) ================================================================================== #
    parser.add_argument('--vl-model', type=str, default='clip', choices=('clip', 'farl'), help="Vision-Language model")
    parser.add_argument('--corpus', type=str, required=True, choices=SEMANTIC_DIPOLES_CORPORA.keys(),
                        help="choose corpus of semantic dipoles")
    parser.add_argument('--vl-sim', type=str, default='proposed-warping',
                        choices=('standard', 'proposed-no-warping', 'proposed-warping'),
                        help="type of VL similarity ('standard', 'proposed-no-warping', 'proposed-warping')")
    parser.add_argument('--no-proj', action='store_true', help="do not project to tangent spaces")
    parser.add_argument('--include-cls-in-mean', action='store_true',
                        help="include the CLS token into calculating text features statistics")
    parser.add_argument('--gamma-0', type=float, default=1.0, help="initial gamma parameter")
    # === Latent Support Sets (LSS) ================================================================================== #
    parser.add_argument('--tied', action='store_true', help="set support set dipoles in tied mode")
    parser.add_argument('--num-latent-support-dipoles', type=int, help="number of latent support dipoles / support set")
    parser.add_argument('--min-shift-magnitude', type=float, default=0.1, help="minimum latent shift magnitude")
    parser.add_argument('--max-shift-magnitude', type=float, default=0.2, help="maximum latent shift magnitude")

    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=10000, help="maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, required=True, help="training batch size -- this should be less than "
                                                                      "or equal to the size of the given corpus")
    parser.add_argument('--temperature', type=float, default=1.0, help="contrastive temperature")
    parser.add_argument('--lr', type=float, default=1e-3, help="latent support sets (LSS) learning rate")
    parser.add_argument('--id', action='store_true', help="impose ID preservation using ArcFace loss")
    parser.add_argument('--lambda-id', type=float, default=1000, help="ID loss weighting parameter")
    parser.add_argument('--log-freq', default=10, type=int, help='number of iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='number of iterations per checkpoint model saving')

    # === CUDA ======================================================================================================= #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Check given VL model
    if args.vl_model == 'farl' and args.gan not in ('pggan_celebahq1024', 'stylegan2_ffhq1024'):
        raise ValueError("Invalid VL model ({}) for the given GAN type ({})".format(args.vl_model, args.gan))

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

    # Load CLIP model
    clip_model = None
    if args.vl_model == 'clip':
        clip_model, _ = clip.load("ViT-B/32", device='cuda' if use_cuda else 'cpu', jit=False)
    # Load FaRL model
    elif args.vl_model == 'farl':
        clip_model, _ = clip.load("ViT-B/16", device='cpu')
        if use_cuda:
            clip_model = clip_model.to('cuda')
        farl_model = osp.join('models', 'pretrained', 'farl', FARL_PRETRAIN_MODEL)
        print("  \\__Loading FaRL model: {}".format(farl_model))
        farl_state = torch.load(farl_model)
        clip_model.load_state_dict(farl_state["state_dict"], strict=False)
    clip_model.float()
    clip_model.eval()

    # Get VL text features, means and covariances calculated on the tokens for the semantic dipoles of the given corpus
    sd = SemanticDipoles(corpus=SEMANTIC_DIPOLES_CORPORA[args.corpus],
                         clip_model=clip_model,
                         include_cls_in_mean=args.include_cls_in_mean,
                         use_cuda=use_cuda)
    semantic_dipoles_cls, semantic_dipoles_means, semantic_dipoles_covariances = sd.get_dipole_features()

    # Experiment preprocessing
    exp_preprocessor = ExpPreprocess(gan_type=args.gan, gan_generator=G, stylegan_space=args.stylegan_space,
                                     stylegan_layer=args.stylegan_layer, truncation=args.truncation, exp_dir=exp_dir,
                                     use_cuda=use_cuda, verbose=True)

    # Calculate latent codes centres anf Jung radii for given latent space
    latent_centres, jung_radii = exp_preprocessor.calculate_latent_centres_and_jung_radii()

    # Build Corpus Support Sets model CSS
    print("#. Build Corpus Support Sets CSS...")
    print("  \\__Number of corpus support sets    : {}".format(sd.num_dipoles))
    print("  \\__Number of corpus support dipoles : {}".format(1))
    print("  \\__args.include_cls_in_mean         : {}".format(args.include_cls_in_mean))
    print("  \\__Prompt features dim              : {}".format(sd.dim))
    print("  \\__RBF gamma_0                      : {}".format(args.gamma_0))
    print("  \\__Vision-Language similarity       : {}".format(args.vl_sim))

    CSS = CorpusSupportSets(semantic_dipoles_cls=semantic_dipoles_cls,
                            semantic_dipoles_means=semantic_dipoles_means,
                            semantic_dipoles_covariances=semantic_dipoles_covariances,
                            gamma_0=args.gamma_0)

    # Count number of trainable parameters
    CSS_trainable_parameters = sum(p.numel() for p in CSS.parameters() if p.requires_grad)
    print("  \\__Trainable parameters             : {:,}".format(CSS_trainable_parameters))

    # Build Latent Support Sets model LSS
    print("#. Build Latent Support Sets LSS model(s)...".format(len(latent_centres)))
    print("  \\__Number of latent support sets    : {}".format(sd.num_dipoles))
    print("  \\__Number of latent support dipoles : {}".format(args.num_latent_support_dipoles))
    print("  \\__Support Vectors dim              : {}".format(G.dim_z))
    print("  \\__Learning rate                    : {}".format(args.lr))

    LSS = []
    for lss_idx in range(len(latent_centres)):
        LSS.append(LatentSupportSets(num_support_sets=sd.num_dipoles,
                                     num_support_dipoles=args.num_latent_support_dipoles,
                                     support_vectors_dim=G.dim_z,
                                     latent_centre=latent_centres[lss_idx],
                                     jung_radius=jung_radii[lss_idx]))

    # Count number of trainable parameters
    LSS_trainable_parameters = 0
    for lss_idx in range(len(latent_centres)):
        LSS_trainable_parameters += sum(p.numel() for p in LSS[lss_idx].parameters() if p.requires_grad)
    print("  \\__Trainable parameters             : {:,}".format(LSS_trainable_parameters))

    # Build ID loss (ArcFace)
    if args.id:
        print("#. Build ArcFace (ID loss)...")
    id_loss = IDLoss()

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    t = Trainer(params=args, exp_dir=exp_dir, exp_subdir=args.exp_subdir, use_cuda=use_cuda, multi_gpu=multi_gpu)

    # Train
    t.train(generator=G, latent_support_sets=LSS, corpus_support_sets=CSS, clip_model=clip_model,
            id_loss=id_loss if args.id else None)


if __name__ == '__main__':
    main()
