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
            ["a picture of a female face.",
             "a picture of a male face."],
            ["a picture of an old person.",
             "a picture of a young person."],
            ["a picture of a smiling face.",
             "a picture of a face in neutral expression."],
            ["a picture of a person with red hair.",
             "a picture of a person with black hair."],
            ["a picture of a bald man.",
             "a picture of a man with hair."],
            ["a picture of a man with beard.",
             "a picture of a shaved man."],
            ["a picture of a face with makeup.",
             "a picture of a face without makeup."],
            ["a picture of a person with closed eyes.",
             "a picture of a person with open eyes."],
            ["a picture of a person with tanned skin.",
             "a picture of a person with pale skin."],
        ],

    'expressions':
        [
            ["a picture of a person with happy face.",
             "a picture of a face in neutral expression."],
            ["a picture of a person with sad face.",
             "a picture of a face in neutral expression."],
            ["a picture of a person with angry face.",
             "a picture of a face in neutral expression."],
            ["a picture of a person with surprised face.",
             "a picture of a face in neutral expression."],
            ["a picture of a person with disgusted face.",
             "a picture of a face in neutral expression."],
            ["a picture of a person with fearful face.",
             "a picture of a face in neutral expression."],
            ["a picture of a person with happy face.",
             "a picture of a person with sad face."],
            ["a picture of a person with happy face.",
             "a picture of a person with angry face."],
            ["a picture of a person with surprised face.",
             "a picture of a person with angry face."],
            ["a picture of a person with surprised face.",
             "a picture of a person with happy face."],
            ["a picture of a person with fearful face.",
             "a picture of a person with happy face."],
        ],

    'expressions3':
        [
            ["a picture of a person with happy face.",
             "a picture of a person with angry face."],
            ["a picture of a person with surprised face.",
             "a picture of a person with angry face."],
            ["a picture of a person with surprised face.",
             "a picture of a person with happy face."],
        ],
    'complex':
        [
            ["a picture of a man with a beard crying.",
             "a picture of an angry shaved man."],
            ["a picture of a man with a beard crying.",
             "a picture of a happy shaved man."],
            ["a picture of a man with a beard crying.",
             "a picture of a shaved man with makeup."],
        ],
    'dogs':
        [
            ["a picture of a happy dog.",
             "a picture of a sad dog."],
            ["a picture of a pointy eared dog.",
             "a picture of a droopy eared dog."],
            ["a picture of a long haired dog.",
             "a picture of a short haired dog."],
            ["a picture of a friendly dog.",
             "a picture of an aggressive dog."],
            ["a picture of a dog with big eyes.",
             "a picture of a dog with small eyes."]
        ],

    'cats':
        [
            ["a picture of a long haired cat.",
             "a picture of a short haired cat."],
            ["a picture of a cute cat.",
             "a picture of an ugly cat."],
            ["a picture of a cat with big ears.",
             "a picture of a cat with small ears."]
        ],

    'cars':
        [
            ["a picture of a jeep.",
             "a picture of a low car."],
            ["a picture of a sports car.",
             "a picture of a city car."],
            ["a picture of a modern car.",
             "a picture of a car from the sixties."],
        ],
}


########################################################################################################################
##                                                                                                                    ##
##                                          [ Pre-trained ContraCLIP models ]                                         ##
##                                                                                                                    ##
########################################################################################################################
ContraCLIP_models = (
    'https://www.dropbox.com/s/b8msg3k6zkr0978/contraclip_models.tar?dl=1',
    'fa74df75d074d44059b5cdd1accca8269e6c507dfff43e7cb3ed223f64d78174'
)


########################################################################################################################
##                                                                                                                    ##
##                                                    [ ArcFace ]                                                     ##
##                                                                                                                    ##
########################################################################################################################
ARCFACE = ('https://www.dropbox.com/s/idulblr8pdrmbq1/arcface.tar?dl=1',
           'edd5854cacd86c17a78a11f70ab8c49bceffefb90ee070754288fa7ceadcdfb2')


########################################################################################################################
##                                                                                                                    ##
##                                             [ GenForce GAN Generators ]                                            ##
##                                                                                                                    ##
########################################################################################################################
GENFORCE = ('https://www.dropbox.com/s/96mzot8xap4k0c2/genforce.tar?dl=1',
            '498a7f0701246aeb2a4ee99ebf0dc8d05a8380324cc1385f7e667e6409013670')

GENFORCE_MODELS = {
    # ===[ ProgGAN ]===
    'pggan_bird256': ('pggan_bird256.pth', 256),
    'pggan_car256': ('pggan_car256.pth', 256),
    'pggan_celebahq1024': ('pggan_celebahq1024.pth', 1024),
    'pggan_church256': ('pggan_church256.pth', 256),
    'pggan_pottedplant256': ('pggan_pottedplant256.pth', 256),
    # ===[ StyleGAN1 ]===
    'stylegan_animeface512': ('stylegan_animeface512.pth', 512),
    'stylegan_animeportrait512': ('stylegan_animeportrait512.pth', 512),
    'stylegan_apartment256': ('stylegan_apartment256.pth', 256),
    'stylegan_artface512': ('stylegan_artface512.pth', 512),
    'stylegan_bedroom256': ('stylegan_bedroom256.pth', 256),
    'stylegan_car512': ('stylegan_car512.pth', 512),
    'stylegan_cat256': ('stylegan_cat256.pth', 256),
    'stylegan_celebahq1024': ('stylegan_celebahq1024.pth', 1024),
    'stylegan_ffhq1024': ('stylegan_ffhq1024.pth', 1024),
    'stylegan_tower256': ('stylegan_tower256.pth', 256),
    # ===[ StyleGAN2 ]===
    'stylegan2_afhqcat512': ('stylegan2_afhqcat512.pth', 512),
    'stylegan2_afhqdog512': ('stylegan2_afhqdog512.pth', 512),
    'stylegan2_afhqv2512': ('stylegan2_afhqv2512.pth', 512),
    'stylegan2_car512': ('stylegan2_car512.pth', 512),
    'stylegan2_cat256': ('stylegan2_cat256.pth', 256),
    'stylegan2_church256': ('stylegan2_church256.pth', 256),
    'stylegan2_ffhq1024': ('stylegan2_ffhq1024.pth', 1024),
    'stylegan2_horse256': ('stylegan2_horse256.pth', 256)
}
