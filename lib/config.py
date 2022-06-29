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
            ["a photo of a female.",
             "a photo of a male."],
            ["a photo of an old person.",
             "a photo of a young person."],
            ["a photo of a smiling person.",
             "a photo of a sad person."],
            ["a photo of a bald man.",
             "a photo of a man with hair."],
            ["a photo of a man with beard.",
             "a photo of a shaved man."],
            ["a photo of a face with makeup.",
             "a photo of a face without makeup."],
            ["a photo of a person with closed eyes.",
             "a photo of a person with open eyes."],
            ["a photo of a person with open lips.",
             "a photo of a person with closed lips."],
            ["a photo of a person with tanned skin.",
             "a photo of a person with pale skin."],
        ],
    'expressions':
        [
            ["a photo of a person with a happy face.",
             "a photo of a person with a neutral face."],
            ["a photo of a person with a sad face.",
             "a photo of a person with a neutral face."],
            ["a photo of a person with a fearful face.",
             "a photo of a person with a neutral face."],
            ["a photo of a person with a disgusted face.",
             "a photo of a person with a neutral face."],
            ["a photo of a person with an angry face.",
             "a photo of a person with a neutral face."],
            ["a photo of a person in surprise.",
             "a photo of a person with a neutral face."],
            ["a photo of a person with a sad face.",
             "a photo of a person with a happy face."],
            ["a photo of a person with a fearful face.",
             "a photo of a person with a happy face."],
            ["a photo of a person with a disgusted face.",
             "a photo of a person with a happy face."],
            ["a photo of a person with an angry face.",
             "a photo of a person with a happy face."],
            ["a photo of a person in surprise.",
             "a photo of a person with a happy face."],
            ["a photo of a person with a fearful face.",
             "a photo of a person with a sad face."],
            ["a photo of a person with a disgusted face.",
             "a photo of a person with a sad face."],
            ["a photo of a person with an angry face.",
             "a photo of a person with a sad face."],
            ["a photo of a person in surprise.",
             "a photo of a person with a sad face."],
            ["a photo of a person with a disgusted face.",
             "a photo of a person with a fearful face."],
            ["a photo of a person with an angry face.",
             "a photo of a person with a fearful face."],
            ["a photo of a person in surprise.",
             "a photo of a person with a fearful face."],
            ["a photo of a person with an angry face.",
             "a photo of a person with a disgusted face."],
            ["a photo of a person in surprise.",
             "a photo of a person with a disgusted face."],
            ["a photo of a person in surprise.",
             "a photo of a person with an angry face."],
        ],
    'expressions3':
        [
            ["a photo of a person with a happy face.",
             "a photo of a person with an angry face."],
            ["a photo of a person in surprise.",
             "a photo of a person with an angry face."],
            ["a photo of a person in surprise.",
             "a photo of a person with a happy face."],
        ],
    'complex':
        [
            ["a photo of a man with a beard crying.",
             "a photo of an angry shaved man."],
            ["a photo of a man with a beard crying.",
             "a photo of a happy shaved man."],
            ["a photo of a man with a beard crying.",
             "a photo of a shaved man with makeup."],
        ],
    'dogs':
        [
            ["a photo of a happy dog.",
             "a photo of a sad dog."],
            ["a photo of a long haired dog.",
             "a photo of a short haired dog."],
            ["a photo of a friendly dog.",
             "a photo of an aggressive dog."],
            ["a photo of a dog with big eyes.",
             "a photo of a dog with small eyes."]
        ],
    'cats':
        [
            ["a photo of a long haired cat.",
             "a photo of a short haired cat."],
            ["a photo of a cute cat.",
             "a photo of an ugly cat."],
            ["a photo of a cat with big ears.",
             "a photo of a cat with small ears."]
        ],
    'cars':
        [
            ["a photo of a jeep.",
             "a photo of a low car."],
            ["a photo of a sports car.",
             "a photo of a city car."],
            ["a photo of a modern car.",
             "a photo of a car from the sixties."],
        ],
}


########################################################################################################################
##                                                                                                                    ##
##                                          [ Pre-trained ContraCLIP models ]                                         ##
##                                                                                                                    ##
########################################################################################################################
ContraCLIP_models = ('https://www.dropbox.com/s/b8msg3k6zkr0978/contraclip_models.tar?dl=1',
                     '5813e2e6b9bb44cf3e9c59b33b575fd906a50ee1d2a0a2be61b23798537de883')

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
