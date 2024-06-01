from .cnn1 import CNNNetwork1
from .cnn2 import CNNNetwork2
from .vgg16 import VGG16Network
from .vgg19 import VGG19Network

__all__ = ["CNNNetwork1", "CNNNetwork2", "VGG16Network", "VGG19Network"]

SUPPORTED_MODELS = ["CNN1", "CNN2", "VGG16", "VGG19"]
