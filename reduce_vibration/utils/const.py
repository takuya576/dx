from torchvision import models

# config.netに対応するモデル名とモデルクラスの対応を定義した辞書
model_mapping = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "densenet121": models.densenet121,
    "vgg19_bn": models.vgg19_bn,
    # 他のモデルを追加することもできます
}
