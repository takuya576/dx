from torchvision import models

# config.netに対応するモデル名とモデルクラスの対応を定義した辞書
model_mapping = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "resnet152": models.resnet152,
    "densenet121": models.densenet121,
    "vgg16_bn": models.vgg16_bn,
    "vgg19_bn": models.vgg19_bn,
    "vit_b_16": models.vit_b_16,
    "vit_l_16": models.vit_l_16,
    # 他のモデルを追加することもできます
}
