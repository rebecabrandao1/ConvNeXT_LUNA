import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from transformers import ConvNextV2Model

class ConvNextV2FPNBackbone(nn.Module):
    def __init__(self, model_name="facebook/convnextv2-tiny-1k-224"):
        super().__init__()
        # Carrega o corpo do modelo (Backbone) com GRN integrada [cite: 705, 1254]
        self.convnext = ConvNextV2Model.from_pretrained(model_name)
        
        # Canais de saída dos 4 estágios do ConvNeXt V2 Tiny [cite: 1254]
        self.stage_channels = self.convnext.config.hidden_sizes # [96, 192, 384, 768]
        self.out_channels = 256 # Canal padrão para a FPN [cite: 707]

        # Feature Pyramid Network para detecção multiescala [cite: 1254]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.stage_channels,
            out_channels=self.out_channels
        )

    def forward(self, x):
        # Extrai os hidden states dos 4 estágios [cite: 1254]
        outputs = self.convnext(x, output_hidden_states=True)
        
        # Mapeia as saídas para o dicionário esperado pela FPN
        # hidden_states[1] a [4] correspondem aos estágios de resolução 1/4 a 1/32
        features = {
            "0": outputs.hidden_states[1],
            "1": outputs.hidden_states[2],
            "2": outputs.hidden_states[3],
            "3": outputs.hidden_states[4]
        }
        
        return self.fpn(features)

def create_mask_rcnn_model(num_classes=2):
    """
    Configura o Mask R-CNN com âncoras para nódulos pequenos (<15mm)[cite: 1113].
    """
    backbone = ConvNextV2FPNBackbone()
    
    # Gerador de Âncoras ajustado para nódulos pequenos do LUNA16 
    # Usamos tamanhos menores (4 a 16) para garantir a detecção na faixa R3 [cite: 310, 507]
    anchor_sizes = ((4,), (8,), (12,), (16,)) 
    aspect_ratios = ((0.8, 1.0, 1.2),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # Multi-scale ROI Align para preservação de detalhes espaciais [cite: 712, 1254]
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

    # Modelo final Multi-task [cite: 699, 751]
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler
    )
    
    return model