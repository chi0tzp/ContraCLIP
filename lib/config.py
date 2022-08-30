########################################################################################################################
## Basic configuration file.                                                                                          ##
##                                                                                                                    ##
##                                                                                                                    ##
########################################################################################################################


########################################################################################################################
##                                                                                                                    ##
##                                            [ Semantic Dipoles Corpora ]                                            ##
##                                                                                                                    ##
########################################################################################################################
SEMANTIC_DIPOLES_CORPORA = {
    'dev':
        [
            # 0
            ["a photo of an old person.",
             "a photo of a young person."],
            # 1
            ["a photo of a person with makeup.",
             "a photo of a person without makeup."]
        ],
    'attributes':
        [
            # 0
            ["a photo of an old person.",
             "a photo of a young person."],
            # 1
            ["a photo of a female.",
             "a photo of a male."],
            # 2
            ["a photo of a man with beard.",
             "a photo of a shaved man."],
            # 3
            ["a photo of a face with makeup.",
             "a photo of a face without makeup."],
            # 4
            ["a photo of a person with tanned skin.",
             "a photo of a person with pale skin."],
        ],
    'expressions':
        [
            # 0
            ["a photo of a person in surprise.",
             "a photo of a person with a neutral face."],
            # 1
            ["a photo of a person with a disgusted face.",
             "a photo of a person with an angry face."],
            # 2
            ["a photo of a person with a happy face.",
             "a photo of a person with a sad face."],
        ],
    'dogs':
        [
            # 0
            ["a photo of a happy dog.",
             "a photo of a sad dog."],
            # 1
            ["a photo of a long haired dog.",
             "a photo of a short haired dog."],
            # 2
            ["a photo of a friendly dog.",
             "a photo of an aggressive dog."],
            # 3
            ["a photo of a dog with big eyes.",
             "a photo of a dog with small eyes."]
        ],
    'cats':
        [
            # 0
            ["a photo of a long haired cat.",
             "a photo of a short haired cat."],
            # 1
            ["a photo of a cute cat.",
             "a photo of an ugly cat."],
            # 2
            ["a photo of a cat with big ears.",
             "a photo of a cat with small ears."]
        ],
    'cars':
        [
            # 0
            ["a photo of a jeep.",
             "a photo of a low car."],
            # 1
            ["a photo of a sports car.",
             "a photo of a city car."],
            # 2
            ["a photo of a modern car.",
             "a photo of a car from the sixties."],
        ]
}


########################################################################################################################
##                                                                                                                    ##
##                                          [ Pre-trained ContraCLIP models ]                                         ##
##                                                                                                                    ##
########################################################################################################################
# TODO: to appear soon
ContraCLIP_models = ('XXX', 'XXX')

########################################################################################################################
##                                                                                                                    ##
##                                                      [ FaRL ]                                                      ##
##                                                                                                                    ##
########################################################################################################################
# Choose pre-trained FaRL model (epoch 16 or 64)
FARL_EP = 64
FARL_PRETRAIN_MODEL = 'FaRL-Base-Patch16-LAIONFace20M-ep{}.pth'.format(FARL_EP)

FARL = ('https://www.dropbox.com/s/7mj2ec3z565cmh5/farl.tar?dl=1',
        'e15cac0fa698fcaac42e76867e229303b625ccd2c56d5de989cb4af9c6135a3a')

########################################################################################################################
##                                                                                                                    ##
##                                            [ GAN Images CLIP Features ]                                            ##
##                                                                                                                    ##
########################################################################################################################
GAN_CLIP_FEATURES = ('https://www.dropbox.com/s/1sp0le1puivxps5/gan_clip_features.tar?dl=1',
                     '41343f98a62bbeca1d9e1cec160c867b8d381d52e21dc9b1b1ed9728e0cdd352')

########################################################################################################################
##                                                                                                                    ##
##                                                     [ SFD ]                                                        ##
##                                                                                                                    ##
########################################################################################################################
SFD = ('https://www.dropbox.com/s/jssqpwyp4edp20o/sfd.tar?dl=1',
       '2bea5f1c10110e356eef3f4efd45169100b9c7704eb6e6abd309df58f34452d4')

########################################################################################################################
##                                                                                                                    ##
##                                                    [ ArcFace ]                                                     ##
##                                                                                                                    ##
########################################################################################################################
ARCFACE = ('https://www.dropbox.com/s/idulblr8pdrmbq1/arcface.tar?dl=1',
           'edd5854cacd86c17a78a11f70ab8c49bceffefb90ee070754288fa7ceadcdfb2')

########################################################################################################################
##                                                                                                                    ##
##                                                   [ FairFace ]                                                     ##
##                                                                                                                    ##
########################################################################################################################
FAIRFACE = ('https://www.dropbox.com/s/lqrydpw7nv27ass/fairface.tar?dl=1',
            '0e78ff8b79612e52e226461fb67f6cff43cef0959d1ab2b520acdcc9105d065e')

########################################################################################################################
##                                                                                                                    ##
##                                                    [ HopeNet ]                                                     ##
##                                                                                                                    ##
########################################################################################################################
HOPENET = ('https://www.dropbox.com/s/rsw7gmo4gkqrbsv/hopenet.tar?dl=1',
           '8c9d67dd8f82ce3332c43b5fc407dc57674d1f16fbe7f0743e9ad57ede73e33f')

########################################################################################################################
##                                                                                                                    ##
##                                                 [ AU Detector ]                                                    ##
##                                                                                                                    ##
########################################################################################################################
AUDET = ('https://www.dropbox.com/s/jkkf1gda9o8ed47/au_detector.tar?dl=1',
         'dbdf18bf541de3c46769d712866bef38496b7528072850c28207747b2b2c101e')

########################################################################################################################
##                                                                                                                    ##
##                                              [ CelebA Attributes ]                                                 ##
##                                                                                                                    ##
########################################################################################################################
CELEBA_ATTRIBUTES = ('https://www.dropbox.com/s/bxbegherkpvgbw9/celeba_attributes.tar?dl=1',
                     '45276f2df865112c7488fe128d8c79527da252aad30fc541417b9961dfdd9bbc')


########################################################################################################################
##                                                                                                                    ##
##                                                     [ FER ]                                                        ##
##                                                                                                                    ##
########################################################################################################################
FER = ('https://www.dropbox.com/s/1u6e7yvss56nx1n/fer.tar?dl=1',
       '94b1f8c23dfc5e626c1de0e76257f174463b8f2c371670036ce75f0923d4985d')

########################################################################################################################
##                                                                                                                    ##
##                                             [ GenForce GAN Generators ]                                            ##
##                                                                                                                    ##
########################################################################################################################
GENFORCE = ('https://www.dropbox.com/s/3osul10173lbhut/genforce.tar?dl=1',
            '369f13eade75f906ab74dc826b9e9f795fd4137d20f1c8e4e28bb92b5ba8b1a7')

GENFORCE_MODELS = {
    # ===[ ProgGAN ]===
    'pggan_celebahq1024': ('pggan_celebahq1024.pth', 1024),
    # ===[ StyleGAN2 ]===
    'stylegan2_ffhq1024': ('stylegan2_ffhq1024.pth', 1024),
    'stylegan2_afhqcat512': ('stylegan2_afhqcat512.pth', 512),
    'stylegan2_afhqdog512': ('stylegan2_afhqdog512.pth', 512),
    'stylegan2_car512': ('stylegan2_car512.pth', 512),
}

STYLEGAN_LAYERS = {
    'stylegan2_ffhq1024': 18,
    'stylegan2_afhqcat512': 16,
    'stylegan2_afhqdog512': 16,
    'stylegan2_car512': 16,
    'stylegan2_church256': 14,
}

STYLEGAN2_STYLE_SPACE_TARGET_LAYERS = {
    'stylegan2_ffhq1024':
        {
            'style00': 512,  # 'layer0'  : '4x4/Conv'
            'style01': 512,  # 'layer1'  : '8x8/Conv0_up'
            'style02': 512,  # 'layer2'  : '8x8/Conv1'
            'style03': 512,  # 'layer3'  : '16x16/Conv0_up'
            'style04': 512,  # 'layer4'  : '16x16/Conv1'
            'style05': 512,  # 'layer5'  : '32x32/Conv0_up'
            'style06': 512,  # 'layer6'  : '32x32/Conv1'
            'style07': 512,  # 'layer7'  : '64x64/Conv0_up'
            'style08': 512,  # 'layer8'  : '64x64/Conv1'
            'style09': 512,  # 'layer9'  : '128x128/Conv0_up'
            'style10': 256,  # 'layer10' : '128x128/Conv1'
            'style11': 256,  # 'layer11' : '256x256/Conv0_up'
            'style12': 128,  # 'layer12' : '256x256/Conv1'
            'style13': 128,  # 'layer13' : '512x512/Conv0_up'
            'style14': 64,   # 'layer14' : '512x512/Conv1'
            'style15': 64,   # 'layer15' : '1024x1024/Conv0_up'
            'style16': 32    # 'layer16' : '1024x1024/Conv1'
        },
    'stylegan2_afhqcat512':
        {
            'style00': 512,
            'style01': 512,
            'style02': 512,
            'style03': 512,
            'style04': 512,
            'style05': 512,
            'style06': 512,
            'style07': 512,
            'style08': 512,
            'style09': 512,
            'style10': 256,
            'style11': 256,
            'style12': 128,
            'style13': 128,
            'style14': 64,
        },
    'stylegan2_afhqdog512':
        {
            'style00': 512,
            'style01': 512,
            'style02': 512,
            'style03': 512,
            'style04': 512,
            'style05': 512,
            'style06': 512,
            'style07': 512,
            'style08': 512,
            'style09': 512,
            'style10': 256,
            'style11': 256,
            'style12': 128,
            'style13': 128,
            'style14': 64,
        },
    'stylegan2_car512':
        {
            'style00': 512,
            'style01': 512,
            'style02': 512,
            'style03': 512,
            'style04': 512,
            'style05': 512,
            'style06': 512,
            'style07': 512,
            'style08': 512,
            'style09': 512,
            'style10': 256,
            'style11': 256,
            'style12': 128,
            'style13': 128,
            'style14': 64,
        },
    'stylegan2_church256':
        {
            'style00': 512,
            'style01': 512,
            'style02': 512,
            'style03': 512,
            'style04': 512,
            'style05': 512,
            'style06': 512,
            'style07': 512,
            'style08': 512,
            'style09': 512,
            'style10': 256,
            'style11': 256,
            'style12': 128,
        }
}
