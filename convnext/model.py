import torch
from torch import nn
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification

def create_model(num_classes=2, pretrained=True, num_channels=1):
    """
    Cria e adapta um modelo ConvNeXtV2 para a tarefa de classificação.
    """
    if pretrained:
        # Usa modelo pré-treinado do HuggingFace
        model = ConvNextV2ForImageClassification.from_pretrained(
            "facebook/convnextv2-tiny-1k-224",
            num_channels=num_channels,
            ignore_mismatched_sizes=True,
            num_labels=num_classes
        )
    else:
        # Usa implementação local do ConvNeXt-V2
        from ConvNeXtV2.models.convnextv2 import convnextv2_tiny
        model = convnextv2_tiny(
            num_classes=num_classes,
            in_chans=num_channels
        )
    
    print(f"Modelo ConvNeXtV2 criado com {num_classes} classes de saída e {num_channels} canais de entrada.")
    return model

def create_image_processor():
    """
    Cria o processador de imagens.
    """
    return AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")