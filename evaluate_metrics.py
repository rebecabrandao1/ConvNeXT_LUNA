import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d
import torchvision.transforms.functional as F # <-- ADICIONADO AQUI

from convnext.model import create_mask_rcnn_model
from convnext.dataset import create_luna_dataset
from torch.utils.data import DataLoader
import config

def calculate_box_iou(box1, box2):
    """Calcula o IoU entre duas bounding boxes no formato [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_mask_metrics(pred_mask, true_mask):
    """Calcula Dice e IoU para máscaras binárias."""
    pred = pred_mask > 0.5
    true = true_mask > 0.5
    
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    
    iou = intersection / union if union > 0 else 0
    dice = 2 * intersection / (pred.sum() + true.sum()) if (pred.sum() + true.sum()) > 0 else 0
    
    return iou, dice

def collate_fn(batch):
    return tuple(zip(*batch))

# --- O NOSSO TRUQUE (PROCESSADOR FALSO) ---
class CustomProcessor:
    def __call__(self, img, return_tensors="pt"):
        # Faz exatamente o que o predict.py fez, mantendo a imagem de 0 a 1!
        return {"pixel_values": F.to_tensor(img).unsqueeze(0)}
# ------------------------------------------

def evaluate_model(model_path, data_folder, device_name=config.DEVICE, iou_thresh=0.5):
    device = torch.device(device_name)
    print(f"Carregando modelo e avaliando no dispositivo: {device}")
    
    model = create_mask_rcnn_model(num_classes=config.NUM_LABELS)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"Pesos do modelo '{model_path}' carregados com sucesso!")

    model = model.to(device)
    
    # --- USANDO O NOSSO PROCESSADOR FALSO AQUI ---
    processor = CustomProcessor()
    ds = create_luna_dataset(folder=data_folder, image_processor=processor, annotation_format='auto')
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    # ---------------------------------------------
    
    total_gt_nodules = 0
    total_images = len(ds)
    all_predictions = []  
    
    mask_ious = []
    mask_dices = []
    
    print(f"Iniciando inferência em {total_images} imagens...")
    
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Avaliando"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_masks = output['masks'].squeeze(1).cpu().numpy()
                
                gt_boxes = target['boxes'].cpu().numpy()
                gt_masks = target['masks'].squeeze().cpu().numpy() if 'masks' in target else []
                
                if len(gt_masks) > 0 and len(gt_masks.shape) == 2:
                    gt_masks = np.expand_dims(gt_masks, axis=0)
                
                num_gts = len(gt_boxes)
                total_gt_nodules += num_gts
                
                gt_matched = np.zeros(num_gts, dtype=bool)
                
                for p_idx, (p_box, p_score, p_mask) in enumerate(zip(pred_boxes, pred_scores, pred_masks)):
                    is_tp = False
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        iou = calculate_box_iou(p_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_thresh and not gt_matched[best_gt_idx]:
                        is_tp = True
                        gt_matched[best_gt_idx] = True
                        
                        if len(gt_masks) > best_gt_idx:
                            m_iou, m_dice = calculate_mask_metrics(p_mask, gt_masks[best_gt_idx])
                            mask_ious.append(m_iou)
                            mask_dices.append(m_dice)
                    
                    all_predictions.append((p_score, is_tp))
                    
    print("\n--- Calculando Métricas ---")
    
    all_predictions.sort(key=lambda x: x[0], reverse=True)
    
    tps = np.cumsum([1 if p[1] else 0 for p in all_predictions])
    fps = np.cumsum([0 if p[1] else 1 for p in all_predictions])
    
    precisions = tps / (tps + fps + 1e-16)
    recalls = tps / (total_gt_nodules + 1e-16)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-16)
    
    best_f1_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else -1
    best_f1 = f1_scores[best_f1_idx] if best_f1_idx >= 0 else 0
    best_precision = precisions[best_f1_idx] if best_f1_idx >= 0 else 0
    best_recall = recalls[best_f1_idx] if best_f1_idx >= 0 else 0
    
    ap50 = np.trapz(precisions[::-1], recalls[::-1]) if len(precisions) > 0 else 0
    
    avg_mask_iou = np.mean(mask_ious) if len(mask_ious) > 0 else 0
    avg_mask_dice = np.mean(mask_dices) if len(mask_dices) > 0 else 0
    
    fps_per_scan = fps / total_images
    luna_fp_points = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    
    if len(fps_per_scan) > 0:
        froc_interp = interp1d(fps_per_scan, recalls, kind='previous', bounds_error=False, fill_value=(0, recalls[-1]))
        sensitivities_at_luna_fp_points = froc_interp(luna_fp_points)
        lambda_metric = np.mean(sensitivities_at_luna_fp_points)
    else:
        sensitivities_at_luna_fp_points = np.zeros_like(luna_fp_points)
        lambda_metric = 0

    print("=======================================")
    print(" 1. MÉTRICAS DE DETECÇÃO (Bounding Box)")
    print("=======================================")
    print(f" mAP@0.50     : {abs(ap50):.4f}")
    print(f" Sensibilidade: {best_recall:.4f}")
    print(f" Precisão     : {best_precision:.4f}")
    print(f" F1-Score     : {best_f1:.4f}")
    print("\n=======================================")
    print(" 2. MÉTRICAS DE SEGMENTAÇÃO (Máscara)")
    print("=======================================")
    print(f" DSC (Dice)   : {avg_mask_dice:.4f}")
    print(f" Mask IoU     : {avg_mask_iou:.4f}")
    print("\n=======================================")
    print(" 3. MÉTRICAS CLÍNICAS (LUNA16)")
    print("=======================================")
    print(f" Lambda (Λ)   : {lambda_metric:.4f}")
    for fp_val, sens in zip(luna_fp_points, sensitivities_at_luna_fp_points):
        print(f"   Sensibilidade @ {fp_val} FPs/scan = {sens:.4f}")
        
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(recalls, precisions, 'b-', label=f'mAP@0.5 = {abs(ap50):.3f}')
    plt.xlabel('Sensibilidade (Recall)')
    plt.ylabel('Precisão')
    plt.title('Curva Precision-Recall')
    plt.legend(loc='lower left')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(fps_per_scan, recalls, 'r-')
    plt.plot(luna_fp_points, sensitivities_at_luna_fp_points, 'ko', label=f'Lambda (Λ) = {lambda_metric:.3f}')
    plt.xscale('log', base=2)
    plt.xticks(luna_fp_points, [str(v) for v in luna_fp_points])
    plt.xlabel('Falsos Positivos por Scan (FPs/Scan)')
    plt.ylabel('Sensibilidade (Recall)')
    plt.title('Curva FROC (LUNA16)')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plot_file = 'resultado_metricas.png'
    plt.savefig(plot_file)
    print(f"\nGráficos de avaliação salvos em: {plot_file}")

if __name__ == '__main__':
    modelo = 'outputs/detector_epoch_400.pth'
    pastas_teste = {
        "10-15mm": "dataset/dataset_10-15mm_test",
        "15mm": "dataset/dataset_15mm_test",
        "6-10mm": "dataset/dataset_6-10mm_test",
        "6mm": "dataset/dataset_6mm_test"
    }

    for name, folder in pastas_teste.items():
        print(f"\n=== Avaliando conjunto: {name} -> {folder} ===")
        evaluate_model(modelo, folder)