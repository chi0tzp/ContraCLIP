import sys
import os
import os.path as osp
import argparse
import hashlib
import tarfile
import time
import urllib.request
from lib import GENFORCE, GENFORCE_MODELS, SFD, ARCFACE, FAIRFACE, HOPENET, AUDET, CELEBA_ATTRIBUTES, ContraCLIP_models


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r      \\__%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))

    sys.stdout.flush()


def download(src, sha256sum, dest):
    tmp_tar = osp.join(dest, ".tmp.tar")
    try:
        urllib.request.urlretrieve(src, tmp_tar, reporthook)
    except:
        raise ConnectionError("Error: {}".format(src))

    sha256_hash = hashlib.sha256()
    with open(tmp_tar, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

        sha256_check = sha256_hash.hexdigest() == sha256sum
        print()
        print("      \\__Check sha256: {}".format("OK!" if sha256_check else "Error"))
        if not sha256_check:
            raise Exception("Error: Invalid sha256 sum: {}".format(sha256_hash.hexdigest()))

    tar_file = tarfile.open(tmp_tar, mode='r')
    tar_file.extractall(dest)
    os.remove(tmp_tar)


def main():
    """Download pre-trained GAN generators and various pre-trained detectors (used only during testing), as well as
    pre-trained ContraCLIP models:
    -- GenForce GAN generators [1]
    -- SFD face detector [2]
    -- ArcFace [3]
    -- FairFace [4]
    -- Hopenet [5]
    -- AU detector [6] for 12 DISFA [7] Action Units
    -- Facial attributes detector [8] for 5 CelebA [9] attributes
    -- ContraCLIP [10] pre-trained models:
        StyleGAN2@FFHQ
        ProgGAN@CelebA-HQ:
        StyleGAN2@AFHQ-Cats
        StyleGAN2@AFHQ-Dogs
        StyleGAN2@AFHQ-Cars

    References:
         [1] https://genforce.github.io/
         [2] Zhang, Shifeng, et al. "S3FD: Single shot scale-invariant face detector." Proceedings of the IEEE
             international conference on computer vision. 2017.
         [3] Deng, Jiankang, et al. "Arcface: Additive angular margin loss for deep face recognition."
             Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
         [4] Karkkainen, Kimmo, and Jungseock Joo. "FairFace: Face attribute dataset for balanced race, gender, and age."
             arXiv preprint arXiv:1908.04913 (2019).
         [5] Doosti, Bardia, et al. "Hope-net: A graph-based model for hand-object pose estimation." Proceedings of the
             IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
         [6] Ntinou, Ioanna, et al. "A transfer learning approach to heatmap regression for action unit intensity
             estimation." IEEE Transactions on Affective Computing (2021).
         [7] Mavadati, S. Mohammad, et al. "DISFA: A spontaneous facial action intensity database." IEEE Transactions on
             Affective Computing 4.2 (2013): 151-160.
         [8] Jiang, Yuming, et al. "Talk-to-Edit: Fine-Grained Facial Editing via Dialog." Proceedings of the IEEE/CVF
             International Conference on Computer Vision. 2021.
         [9] Liu, Ziwei, et al. "Deep learning face attributes in the wild." Proceedings of the IEEE international
             conference on computer vision. 2015.
        [10] Tzelepis, C., Oldfield, J., Tzimiropoulos, G., & Patras, I. (2022). ContraCLIP: Interpretable GAN
             generation driven by pairs of contrasting sentences. arXiv preprint arXiv:2206.02104.
    """
    parser = argparse.ArgumentParser(description="Download pre-trained models")
    parser.add_argument('-m', '--contraclip-models', action='store_true', help="download pre-trained ContraCLIP models")
    args = parser.parse_args()

    # Create pre-trained models root directory
    pretrained_models_root = osp.join('models', 'pretrained')
    os.makedirs(pretrained_models_root, exist_ok=True)

    # Download the following pre-trained GAN generators (under models/pretrained/)
    print("#. Download pre-trained GAN generators...")
    print("  \\__.GenForce")
    download_genforce_models = False
    for k, v in GENFORCE_MODELS.items():
        if not osp.exists(osp.join(pretrained_models_root, 'genforce', v[0])):
            download_genforce_models = True
            break
    if download_genforce_models:
        download(src=GENFORCE[0], sha256sum=GENFORCE[1], dest=pretrained_models_root)
    else:
        print("      \\__Already exists.")

    print("#. Download pre-trained ArcFace model...")
    print("  \\__.ArcFace")
    if osp.exists(osp.join(pretrained_models_root, 'arcface', 'model_ir_se50.pth')):
        print("      \\__Already exists.")
    else:
        download(src=ARCFACE[0], sha256sum=ARCFACE[1], dest=pretrained_models_root)

    print("#. Download pre-trained SFD face detector model...")
    print("  \\__.Face detector (SFD)")
    if osp.exists(osp.join(pretrained_models_root, 'sfd', 's3fd-619a316812.pth')):
        print("      \\__Already exists.")
    else:
        download(src=SFD[0], sha256sum=SFD[1], dest=pretrained_models_root)

    print("#. Download pre-trained FairFace model...")
    print("  \\__.FairFace")
    if osp.exists(osp.join(pretrained_models_root, 'fairface', 'fairface_alldata_4race_20191111.pt')) and \
            osp.exists(osp.join(pretrained_models_root, 'fairface', 'res34_fair_align_multi_7_20190809.pt')):
        print("      \\__Already exists.")
    else:
        download(src=FAIRFACE[0], sha256sum=FAIRFACE[1], dest=pretrained_models_root)

    print("#. Download pre-trained Hopenet model...")
    print("  \\__.Hopenet")
    if osp.exists(osp.join(pretrained_models_root, 'hopenet', 'hopenet_alpha1.pkl')) and \
            osp.exists(osp.join(pretrained_models_root, 'hopenet', 'hopenet_alpha2.pkl')) and \
            osp.exists(osp.join(pretrained_models_root, 'hopenet', 'hopenet_robust_alpha1.pkl')):
        print("      \\__Already exists.")
    else:
        download(src=HOPENET[0], sha256sum=HOPENET[1], dest=pretrained_models_root)

    print("#. Download pre-trained AU detector model...")
    print("  \\__.FANet")
    if osp.exists(osp.join(pretrained_models_root, 'au_detector', 'disfa_adaptation_f0.pth')):
        print("      \\__Already exists.")
    else:
        download(src=AUDET[0], sha256sum=AUDET[1], dest=pretrained_models_root)

    print("#. Download pre-trained CelebA attributes predictors models...")
    print("  \\__.CelebA")
    if osp.exists(osp.join(pretrained_models_root, 'celeba_attributes', 'eval_predictor.pth.tar')):
        print("      \\__Already exists.")
    else:
        download(src=CELEBA_ATTRIBUTES[0], sha256sum=CELEBA_ATTRIBUTES[1], dest=pretrained_models_root)

    # Download pre-trained ContraCLIP models
    if args.contraclip_models:
        pretrained_contraclip_root = osp.join('experiments', 'complete')
        os.makedirs(pretrained_contraclip_root, exist_ok=True)

        print("#. Download pre-trained ContraCLIP models...")
        print("  \\__.ContraCLIP pre-trained models...")
        download(src=ContraCLIP_models[0],
                 sha256sum=ContraCLIP_models[1],
                 dest=pretrained_contraclip_root)


if __name__ == '__main__':
    main()
