from torchvision import models

from src.models import MMSEN, MMSEN_small


def assemble_mmsen(assembly=None, num_heads=8, out_channels=[1024, 512]):

    if not models:

        ### load the models
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        densenet121 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        ### remove the layers for classification
        v_model = vgg16.features
        d_model = densenet121.features

        assembly = [v_model, d_model]

    ### freeze the feature extraction layers
    for model in assembly:
        for param in model.parameters():
            param.requires_grad = False

    mmsen = MMSEN(assembly, num_heads, out_channels)

    return mmsen


def assemble_mmsen_small():

    ### load the models
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    densenet121 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    ### remove the layers for classification
    v_model = vgg16.features
    d_model = densenet121.features

    ### freeze the feature extraction layers
    for model in [v_model, d_model]:
        for param in model.parameters():
            param.requires_grad = False

    ### initialize MFGM
    assembly = [d_model, v_model]
    num_heads = 4
    out_channels = [1024, 512]

    return MMSEN_small(assembly, num_heads, out_channels)