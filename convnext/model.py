import torch
import torch.nn as nn
from transformers import ConvNextV2Model, AutoImageProcessor, ConvNextV2ForImageClassification
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

class ConvNextV2FPNBackbone(nn.Module):
    def __init__(self, model_name="facebook/convnextv2-tiny-1k-224"):
        super().__init__()
        # Carrega o modelo base do ConvNeXt V2, que JÁ CONTÉM a camada GRN (Garantindo o passo 4)
        self.convnext = ConvNextV2Model.from_pretrained(model_name)
        
        # O ConvNeXt V2-tiny tem 4 estágios. Precisamos saber o número de canais em cada um: 
        # (normalmente 96, 192, 384, 768 para a versão tiny)
        self.stage_channels = self.convnext.config.hidden_sizes
        self.out_channels = 256 # O tamanho de saída padrão esperado pela FPN e pelo Mask R-CNN
        
        # Cria a FPN para receber as features dos 4 estágios
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.stage_channels,
            out_channels=self.out_channels
        )

    def forward(self, x):
        # Forward pass solicitando os estados ocultos (hidden states)
        outputs = self.convnext(x, output_hidden_states=True)
        
        features = {}
        # outputs.hidden_states[1:] contém os 4 mapas espaciais dos estágios
        for i, f in enumerate(outputs.hidden_states[1:]):
            features[str(i)] = f
            
        # Aplica a FPN (Passo 1 do seu planejamento)
        fpn_features = self.fpn(features)
        
        return fpn_features

def create_mask_rcnn_model(num_classes=2):
    """
    Cria a Mask R-CNN acoplando o backbone ConvNeXt V2.
    """
    backbone = ConvNextV2FPNBackbone()
    
    # RPN com âncoras ajustadas para os nódulos LUNA16 (< 15mm)
    # A Mask R-CNN usa 4 mapas da FPN, portanto passamos 4 tuplas
    # 4px (nódulos de 3 a 5mm), 8px (6 a 9mm), 12px (10 a 14mm), 16px (>= 15mm)
    anchor_sizes = ((4,), (8,), (12,), (16,))
    # Aspect ratios mais coerentes com formato geralmente redondo dos nódulos
    aspect_ratios = ((0.8, 1.0, 1.2),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    
    # RoI Align (Passo 2)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )

    # Cria o modelo final Multi-task
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,  # 0 é background, 1 é nódulo
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler
    )
    
    return model

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