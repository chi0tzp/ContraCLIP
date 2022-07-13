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
    'attributes':
        [
            # 0
            ["a photo of a female.",
             "a photo of a male."],
            # 1
            ["a photo of an old person.",
             "a photo of a young person."],
            # 2
            ["a photo of a person with blonde hair.",
             "a photo of a person with black hair"],
            # 3
            ["a photo of a bald man.",
             "a photo of a man with long hair."],
            # 4
            ["a photo of a man with beard.",
             "a photo of a shaved man."],
            # 5
            ["a photo of a face with makeup.",
             "a photo of a face without makeup."],
            # 6
            ["a photo of a person with tanned skin.",
             "a photo of a person with pale skin."],
        ],
    'expressions':
        [
            # 0
            ["a photo of a person with a happy face.",
             "a photo of a person with a neutral face."],
            # 1
            ["a photo of a person with a sad face.",
             "a photo of a person with a neutral face."],
            # 2
            ["a photo of a person with a fearful face.",
             "a photo of a person with a neutral face."],
            # 3
            ["a photo of a person with a disgusted face.",
             "a photo of a person with a neutral face."],
            # 4
            ["a photo of a person with an angry face.",
             "a photo of a person with a neutral face."],
            # 5
            ["a photo of a person in surprise.",
             "a photo of a person with a neutral face."],
            # 6
            ["a photo of a person with a sad face.",
             "a photo of a person with a happy face."],
            # 7
            ["a photo of a person with a fearful face.",
             "a photo of a person with a happy face."],
            # 8
            ["a photo of a person with a disgusted face.",
             "a photo of a person with a happy face."],
            # 9
            ["a photo of a person with an angry face.",
             "a photo of a person with a happy face."],
            # 10
            ["a photo of a person in surprise.",
             "a photo of a person with a happy face."],
            # 11
            ["a photo of a person with a fearful face.",
             "a photo of a person with a sad face."],
            # 12
            ["a photo of a person with a disgusted face.",
             "a photo of a person with a sad face."],
            # 13
            ["a photo of a person with an angry face.",
             "a photo of a person with a sad face."],
            # 14
            ["a photo of a person in surprise.",
             "a photo of a person with a sad face."],
            # 15
            ["a photo of a person with a disgusted face.",
             "a photo of a person with a fearful face."],
            # 16
            ["a photo of a person with an angry face.",
             "a photo of a person with a fearful face."],
            # 17
            ["a photo of a person in surprise.",
             "a photo of a person with a fearful face."],
            # 18
            ["a photo of a person with an angry face.",
             "a photo of a person with a disgusted face."],
            # 19
            ["a photo of a person in surprise.",
             "a photo of a person with a disgusted face."],
            # 20
            ["a photo of a person in surprise.",
             "a photo of a person with an angry face."],
        ],
    'expressions3':
        [
            # 0
            ["a photo of a person with a happy face.",
             "a photo of a person with an angry face."],
            # 1
            ["a photo of a person in surprise.",
             "a photo of a person with an angry face."],
            # 2
            ["a photo of a person in surprise.",
             "a photo of a person with a happy face."],
        ],
    'complex':
        [
            # 0
            ["a photo of a man with a beard crying.",
             "a photo of an angry shaved man."],
            # 1
            ["a photo of a man with a beard crying.",
             "a photo of a happy shaved man."],
            # 2
            ["a photo of a man with a beard crying.",
             "a photo of a shaved man with makeup."],
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
        ],
    'dev':
        [
            ["a photo of a person with a happy face.",
             "a photo of a person with an angry face."],
            ["a photo of a person with a sad face.",
             "a photo of a person with a disgusted face."],
        ]
}


########################################################################################################################
##                                                                                                                    ##
##                                          [ Pre-trained ContraCLIP models ]                                         ##
##                                                                                                                    ##
########################################################################################################################
ContraCLIP_models = ('https://www.dropbox.com/s/bootpdxhnp9z6ce/contraclip_models.tar?dl=1',
                     '0941c96d311700ef881bed38350d6d0cc38151255a34db94a5f9400758398a7f')

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
##                                             [ GenForce GAN Generators ]                                            ##
##                                                                                                                    ##
########################################################################################################################
GENFORCE = ('https://www.dropbox.com/s/3osul10173lbhut/genforce.tar?dl=1',
            'f9a0f98435cac4fb7599c2cc29858e48365c0998f9f48079efa5faf6c07aa3e1')

GENFORCE_MODELS = {
    # ===[ ProgGAN ]===
    'pggan_celebahq1024': ('pggan_celebahq1024.pth', 1024),
    'pggan_church256': ('pggan_church256.pth', 256),
    'pggan_car256': ('pggan_car256.pth', 256),
    # ===[ StyleGAN2 ]===
    'stylegan2_ffhq1024': ('stylegan2_ffhq1024.pth', 1024),
    'stylegan2_afhqcat512': ('stylegan2_afhqcat512.pth', 512),
    'stylegan2_afhqdog512': ('stylegan2_afhqdog512.pth', 512),
    'stylegan2_car512': ('stylegan2_car512.pth', 512),
    'stylegan2_church256': ('stylegan2_church256.pth', 256)
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
