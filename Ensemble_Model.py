import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
import time
warnings.filterwarnings('ignore')

"""
Triple Ensemble Script for Price Prediction
Combines: CLIP-ViT-H-14 (LAION) + CLIP-ViT-Large (OpenAI) + EfficientNet-BERT
Optimized for A100 GPU with ~70GB memory usage
"""

# ==================== CONFIGURATIONS ====================

class CLIPConfig_H14:
    TRAIN_CSV_PATH = "/home/kumaraa_iitp/hatter/dataset/train.csv"
    TEST_CSV_PATH = "/home/kumaraa_iitp/hatter/dataset/test.csv"
    TRAIN_IMG_DIR = "/home/kumaraa_iitp/kanahia/src/complete_images/train_images"
    TEST_IMG_DIR = "/home/kumaraa_iitp/kanahia/src/complete_images/test_images"
    MODEL_PATH = "/home/kumaraa_iitp/hatter/output2/best_clip_vith_model.pth"
    CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    MAX_TEXT_LENGTH = 77
    BATCH_SIZE = 96 
    VAL_SPLIT = 0.15
    SEED = 42
    NUM_WORKERS = 8

class CLIPConfig_Large:
    TRAIN_CSV_PATH = "/home/kumaraa_iitp/hatter/dataset/train.csv"
    TEST_CSV_PATH = "/home/kumaraa_iitp/hatter/dataset/test.csv"
    TRAIN_IMG_DIR = "/home/kumaraa_iitp/kanahia/src/complete_images/train_images"
    TEST_IMG_DIR = "/home/kumaraa_iitp/kanahia/src/complete_images/test_images"
    MODEL_PATH = "/home/kumaraa_iitp/hatter/output2/best_clip_price_model.pth"
    CLIP_MODEL = "openai/clip-vit-large-patch14"
    MAX_TEXT_LENGTH = 77
    BATCH_SIZE = 128
    VAL_SPLIT = 0.15
    SEED = 42
    NUM_WORKERS = 8

class BERTConfig:
    TRAIN_CSV_PATH = "/home/kumaraa_iitp/hatter/dataset/train.csv"
    TEST_CSV_PATH = "/home/kumaraa_iitp/hatter/dataset/test.csv"
    TRAIN_IMG_DIR = "/home/kumaraa_iitp/kanahia/src/complete_images/train_images"
    TEST_IMG_DIR = "/home/kumaraa_iitp/kanahia/src/complete_images/test_images"
    MODEL_PATH = "/home/kumaraa_iitp/hatter/output/best_efficientnet_bert_model.pth"
    IMAGE_MODEL = "efficientnet_b4"
    TEXT_MODEL = "bert-base-uncased"
    IMAGE_SIZE = 380
    MAX_TEXT_LENGTH = 256
    BATCH_SIZE = 80 
    VAL_SPLIT = 0.15
    SEED = 123
    NUM_WORKERS = 8

# ==================== IMPORT MODEL CLASSES ====================

from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer
from torchvision import models, transforms
from PIL import Image
import os

# CLIP Model (works for both H14 and Large)
class CLIPPricePredictor(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", dropout=0.3):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        embed_dim = self.clip.projection_dim
        
        self.price_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            
            nn.Linear(128, 1)
        )
    
    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        combined = torch.cat([image_embeds, text_embeds], dim=1)
        price = self.price_head(combined)
        return price.squeeze(-1)
    
# CLIP Model (works for both H14 and Large)
class CLIPPricePredictor2(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", dropout=0.3):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        embed_dim = self.clip.projection_dim
        
        self.price_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(128, 1)
        )
    
    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        combined = torch.cat([image_embeds, text_embeds], dim=1)
        price = self.price_head(combined)
        return price.squeeze(-1)

# EfficientNet-BERT Model
class DualEncoderPricePredictor(nn.Module):
    def __init__(self, dropout=0.35):
        super().__init__()
        
        self.image_encoder = models.efficientnet_b4(pretrained=True)
        image_feat_dim = 1792
        self.image_encoder.classifier = nn.Identity()
        
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        text_feat_dim = 768
        
        common_dim = 512
        self.image_projection = nn.Sequential(
            nn.Linear(image_feat_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_feat_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=8,
            dropout=dropout * 0.5,
            batch_first=True
        )
        
        self.fusion_head = nn.Sequential(
            nn.Linear(common_dim * 3, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(192, 1)
        )
    
    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        image_proj = self.image_projection(image_features)
        
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_proj = self.text_projection(text_features)
        
        image_proj_unsqueezed = image_proj.unsqueeze(1)
        text_proj_unsqueezed = text_proj.unsqueeze(1)
        
        attended_features, _ = self.cross_attention(
            image_proj_unsqueezed,
            text_proj_unsqueezed,
            text_proj_unsqueezed
        )
        attended_features = attended_features.squeeze(1)
        
        combined = torch.cat([image_proj, text_proj, attended_features], dim=1)
        log_price = self.fusion_head(combined)
        return log_price.squeeze(-1)

# ==================== DATASET CLASSES ====================

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, image_dir, mode='train'):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.image_dir = image_dir
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['sample_id']
        
        # Load image
        try:
            image_filename = f"{sample_id}.jpg"
            image_path = os.path.join(self.image_dir, image_filename)
            
            if not os.path.exists(image_path):
                for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                    alt_path = os.path.join(self.image_dir, f"{sample_id}{ext}")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
            
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        text = str(row['catalog_content'])
        if pd.isna(text) or text == 'nan':
            text = "No description available"
        
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )
        
        output = {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'sample_id': sample_id
        }
        
        if self.mode == 'train':
            price = float(row['price'])
            log_price = np.log1p(price)
            output['price'] = torch.tensor(log_price, dtype=torch.float32)
            output['original_price'] = torch.tensor(price, dtype=torch.float32)
        
        return output

def get_bert_transforms():
    return transforms.Compose([
        transforms.Resize((BERTConfig.IMAGE_SIZE, BERTConfig.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, image_dir, mode='train'):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.mode = mode
        self.transform = get_bert_transforms()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row['sample_id']
        
        # Load image
        try:
            image_filename = f"{sample_id}.jpg"
            image_path = os.path.join(self.image_dir, image_filename)
            
            if not os.path.exists(image_path):
                for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                    alt_path = os.path.join(self.image_dir, f"{sample_id}{ext}")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
            
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except:
            image = torch.zeros((3, BERTConfig.IMAGE_SIZE, BERTConfig.IMAGE_SIZE))
        
        text = str(row['catalog_content'])
        if pd.isna(text) or text == 'nan':
            text = "No description available"
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=BERTConfig.MAX_TEXT_LENGTH,
            return_tensors='pt'
        )
        
        output = {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'sample_id': sample_id
        }
        
        if self.mode == 'train':
            price = float(row['price'])
            log_price = np.log1p(price)
            output['price'] = torch.tensor(log_price, dtype=torch.float32)
            output['original_price'] = torch.tensor(price, dtype=torch.float32)
        
        return output

# ==================== EVALUATION FUNCTIONS ====================

def calculate_smape(predictions, targets):
    """Calculate SMAPE metric"""
    numerator = np.abs(predictions - targets)
    denominator = (np.abs(targets) + np.abs(predictions)) / 2 + 1e-8
    smape = np.mean(numerator / denominator) * 100
    return smape

def evaluate_model(model, dataloader, device, model_type='clip'):
    """Evaluate model on validation set"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_type.upper()}"):
            if model_type in ['clip_h14', 'clip_large']:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                with torch.cuda.amp.autocast():
                    log_predictions = model(pixel_values, input_ids, attention_mask)
            else:  # bert
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                with torch.cuda.amp.autocast():
                    log_predictions = model(images, input_ids, attention_mask)
            
            # Convert from log space
            predictions = torch.expm1(log_predictions)
            predictions = torch.clamp(predictions, min=0.01)
            
            targets = batch['original_price']
            sample_ids = batch['sample_id']
            
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_sample_ids.extend(sample_ids)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    smape = calculate_smape(all_preds, all_targets)
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mape = np.mean(np.abs((all_targets - all_preds) / all_targets)) * 100
    
    return {
        'sample_ids': all_sample_ids,
        'predictions': all_preds,
        'targets': all_targets,
        'smape': smape,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }

# ==================== ENSEMBLE METHODS ====================

def weighted_ensemble_3(pred1, pred2, pred3, w1=0.45, w2=0.35, w3=0.20):
    """Weighted average for 3 models"""
    return pred1 * w1 + pred2 * w2 + pred3 * w3

def geometric_mean_ensemble_3(pred1, pred2, pred3):
    """Geometric mean for 3 models (good for prices)"""
    return np.power(pred1 * pred2 * pred3, 1/3)

def harmonic_mean_ensemble_3(pred1, pred2, pred3):
    """Harmonic mean for 3 models"""
    return 3 / (1/pred1 + 1/pred2 + 1/pred3 + 1e-8)

def optimized_weighted_ensemble_3(pred1, pred2, pred3, targets):
    """Find optimal weights for 3 models using grid search"""
    best_smape = float('inf')
    best_weights = (0.33, 0.33, 0.34)
    
    # Coarse grid search
    for w1 in np.arange(0.1, 0.9, 0.1):
        for w2 in np.arange(0.1, 0.9 - w1, 0.1):
            w3 = 1 - w1 - w2
            if w3 < 0.05:
                continue
            
            ensemble_pred = weighted_ensemble_3(pred1, pred2, pred3, w1, w2, w3)
            smape = calculate_smape(ensemble_pred, targets)
            
            if smape < best_smape:
                best_smape = smape
                best_weights = (w1, w2, w3)
    
    # Fine grid search around best weights
    w1_best, w2_best, w3_best = best_weights
    for w1 in np.arange(max(0.0, w1_best - 0.1), min(1.0, w1_best + 0.1), 0.02):
        for w2 in np.arange(max(0.0, w2_best - 0.1), min(1.0 - w1, w2_best + 0.1), 0.02):
            w3 = 1 - w1 - w2
            if w3 < 0.01 or w3 > 1:
                continue
            
            ensemble_pred = weighted_ensemble_3(pred1, pred2, pred3, w1, w2, w3)
            smape = calculate_smape(ensemble_pred, targets)
            
            if smape < best_smape:
                best_smape = smape
                best_weights = (w1, w2, w3)
    
    return best_weights, best_smape

def predict_test_set(model, dataloader, device, model_type='clip'):
    """Generate predictions on test set"""
    model.eval()
    
    all_sample_ids = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting with {model_type.upper()}"):
            if model_type in ['clip_h14', 'clip_large']:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                with torch.cuda.amp.autocast():
                    log_predictions = model(pixel_values, input_ids, attention_mask)
            else:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                with torch.cuda.amp.autocast():
                    log_predictions = model(images, input_ids, attention_mask)
            
            predictions = torch.expm1(log_predictions)
            predictions = torch.clamp(predictions, min=0.01)
            
            sample_ids = batch['sample_id']
            
            all_sample_ids.extend(sample_ids)
            all_predictions.extend(predictions.cpu().numpy())
    
    return all_sample_ids, np.array(all_predictions)

# ==================== MAIN TRIPLE ENSEMBLE PIPELINE ====================

def main():
    print("="*70)
    print("TRIPLE ENSEMBLE MODEL EVALUATION AND PREDICTION")
    print("="*70)
    print("\nModels: CLIP-ViT-H-14 (LAION) + CLIP-ViT-Large (OpenAI) + EfficientNet-BERT")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load training data
    print(f"\nLoading data from {CLIPConfig_H14.TRAIN_CSV_PATH}")
    train_df = pd.read_csv(CLIPConfig_H14.TRAIN_CSV_PATH)
    print(f"Total training samples: {len(train_df)}")
    
    # Create validation split (same seed for all)
    print("\nCreating validation split...")
    train_data, val_data = train_test_split(
        train_df, test_size=0.15, random_state=42
    )
    print(f"Validation set: {len(val_data)} samples")
    
    # ==================== LOAD MODELS ====================
    
    print("\n" + "-"*70)
    print("LOADING MODELS")
    print("-"*70)
    
    start_time = time.time()
    
    # Load CLIP-ViT-H-14 (LAION)
    print("\n1. Loading CLIP-ViT-H-14 (LAION)...")
    processor_h14 = CLIPProcessor.from_pretrained(CLIPConfig_H14.CLIP_MODEL)
    clip_h14_model = CLIPPricePredictor(clip_model_name=CLIPConfig_H14.CLIP_MODEL).to(device)
    
    clip_h14_checkpoint = torch.load(CLIPConfig_H14.MODEL_PATH, map_location=device, weights_only=False)
    clip_h14_model.load_state_dict(clip_h14_checkpoint['model_state_dict'])
    print(f"   ✓ CLIP-ViT-H-14 loaded (Val SMAPE: 40.88%)")
    
    # Load CLIP-ViT-Large (OpenAI)
    print("\n2. Loading CLIP-ViT-Large (OpenAI)...")
    processor_large = CLIPProcessor.from_pretrained(CLIPConfig_Large.CLIP_MODEL)
    clip_large_model = CLIPPricePredictor2(clip_model_name=CLIPConfig_Large.CLIP_MODEL).to(device)
    
    clip_large_checkpoint = torch.load(CLIPConfig_Large.MODEL_PATH, map_location=device, weights_only=False)
    clip_large_model.load_state_dict(clip_large_checkpoint['model_state_dict'])
    print(f"   ✓ CLIP-ViT-Large loaded (Val SMAPE: 41.97%)")
    
    # Load EfficientNet-BERT
    print("\n3. Loading EfficientNet-BERT...")
    bert_tokenizer = AutoTokenizer.from_pretrained(BERTConfig.TEXT_MODEL)
    bert_model = DualEncoderPricePredictor().to(device)
    
    bert_checkpoint = torch.load(BERTConfig.MODEL_PATH, map_location=device, weights_only=False)
    bert_model.load_state_dict(bert_checkpoint['model_state_dict'])
    print(f"   ✓ EfficientNet-BERT loaded (Val SMAPE: 43.17%)")
    
    load_time = time.time() - start_time
    print(f"\n✓ All models loaded in {load_time:.1f}s")
    
    # ==================== EVALUATE ON VALIDATION SET ====================
    
    print("\n" + "-"*70)
    print("VALIDATION SET EVALUATION")
    print("-"*70)
    
    eval_start = time.time()
    
    # Evaluate CLIP-ViT-H-14
    print("\nEvaluating CLIP-ViT-H-14...")
    h14_val_dataset = CLIPDataset(val_data, processor_h14, CLIPConfig_H14.TRAIN_IMG_DIR, mode='train')
    h14_val_loader = DataLoader(
        h14_val_dataset,
        batch_size=CLIPConfig_H14.BATCH_SIZE,
        shuffle=False,
        num_workers=CLIPConfig_H14.NUM_WORKERS,
        pin_memory=True
    )
    
    h14_results = evaluate_model(clip_h14_model, h14_val_loader, device, 'clip_h14')
    
    print(f"\nCLIP-ViT-H-14 Results:")
    print(f"  SMAPE: {h14_results['smape']:.2f}%")
    print(f"  MAE: ${h14_results['mae']:.2f}")
    print(f"  RMSE: ${h14_results['rmse']:.2f}")
    
    # Evaluate CLIP-ViT-Large
    print("\nEvaluating CLIP-ViT-Large...")
    large_val_dataset = CLIPDataset(val_data, processor_large, CLIPConfig_Large.TRAIN_IMG_DIR, mode='train')
    large_val_loader = DataLoader(
        large_val_dataset,
        batch_size=CLIPConfig_Large.BATCH_SIZE,
        shuffle=False,
        num_workers=CLIPConfig_Large.NUM_WORKERS,
        pin_memory=True
    )
    
    large_results = evaluate_model(clip_large_model, large_val_loader, device, 'clip_large')
    
    print(f"\nCLIP-ViT-Large Results:")
    print(f"  SMAPE: {large_results['smape']:.2f}%")
    print(f"  MAE: ${large_results['mae']:.2f}")
    print(f"  RMSE: ${large_results['rmse']:.2f}")
    
    # Evaluate EfficientNet-BERT
    print("\nEvaluating EfficientNet-BERT...")
    bert_val_dataset = BERTDataset(val_data, bert_tokenizer, BERTConfig.TRAIN_IMG_DIR, mode='train')
    bert_val_loader = DataLoader(
        bert_val_dataset,
        batch_size=BERTConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=BERTConfig.NUM_WORKERS,
        pin_memory=True
    )
    
    bert_results = evaluate_model(bert_model, bert_val_loader, device, 'bert')
    
    print(f"\nEfficientNet-BERT Results:")
    print(f"  SMAPE: {bert_results['smape']:.2f}%")
    print(f"  MAE: ${bert_results['mae']:.2f}")
    print(f"  RMSE: ${bert_results['rmse']:.2f}")
    
    eval_time = time.time() - eval_start
    print(f"\n✓ Validation evaluation completed in {eval_time:.1f}s")
    
    # ==================== TEST ENSEMBLE METHODS ====================
    
    print("\n" + "-"*70)
    print("TESTING ENSEMBLE METHODS ON VALIDATION SET")
    print("-"*70)
    
    h14_preds = h14_results['predictions']
    large_preds = large_results['predictions']
    bert_preds = bert_results['predictions']
    targets = h14_results['targets']
    
    print("\nTesting different ensemble methods:")
    print("-" * 50)
    
    # 1. Simple Average (1/3 each)
    simple_preds = (h14_preds + large_preds + bert_preds) / 3
    simple_smape = calculate_smape(simple_preds, targets)
    print(f"1. Simple Average (1/3, 1/3, 1/3):     SMAPE = {simple_smape:.2f}%")
    
    # 2. Weighted (based on individual performance)
    init_weighted_preds = weighted_ensemble_3(h14_preds, large_preds, bert_preds, 0.45, 0.35, 0.20)
    init_weighted_smape = calculate_smape(init_weighted_preds, targets)
    print(f"2. Initial Weighted (0.45, 0.35, 0.20): SMAPE = {init_weighted_smape:.2f}%")
    
    # 3. Optimized Weighted
    print("\n3. Finding optimal weights via grid search...")
    opt_weights, opt_weighted_smape = optimized_weighted_ensemble_3(h14_preds, large_preds, bert_preds, targets)
    print(f"   Optimal weights:")
    print(f"     ViT-H-14:    {opt_weights[0]:.3f}")
    print(f"     ViT-Large:   {opt_weights[1]:.3f}")
    print(f"     BERT:        {opt_weights[2]:.3f}")
    print(f"   Optimized Weighted:                SMAPE = {opt_weighted_smape:.2f}%")
    
    # 4. Geometric Mean
    geo_preds = geometric_mean_ensemble_3(h14_preds, large_preds, bert_preds)
    geo_smape = calculate_smape(geo_preds, targets)
    print(f"\n4. Geometric Mean:                      SMAPE = {geo_smape:.2f}%")
    
    # 5. Harmonic Mean
    harm_preds = harmonic_mean_ensemble_3(h14_preds, large_preds, bert_preds)
    harm_smape = calculate_smape(harm_preds, targets)
    print(f"5. Harmonic Mean:                      SMAPE = {harm_smape:.2f}%")
    
    # Determine best method
    methods = {
        'simple_average': simple_smape,
        'optimized_weighted': opt_weighted_smape,
        'geometric_mean': geo_smape,
        'harmonic_mean': harm_smape
    }
    
    best_method = min(methods, key=methods.get)
    best_method_smape = methods[best_method]
    
    print("\n" + "="*50)
    print(f"🏆 BEST ENSEMBLE METHOD: {best_method.upper()}")
    print(f"   SMAPE: {best_method_smape:.2f}%")
    print("="*50)
    
    # ==================== GENERATE TEST PREDICTIONS ====================
    
    print("\n" + "-"*70)
    print("GENERATING TEST SET PREDICTIONS")
    print("-"*70)
# Load test data
    print(f"\nLoading test data from {CLIPConfig_H14.TEST_CSV_PATH}")
    test_df = pd.read_csv(CLIPConfig_H14.TEST_CSV_PATH)
    print(f"Total test samples: {len(test_df)}")
    
    pred_start = time.time()
    
    # Predict with CLIP-ViT-H-14
    print("\nGenerating predictions with CLIP-ViT-H-14...")
    h14_test_dataset = CLIPDataset(test_df, processor_h14, CLIPConfig_H14.TEST_IMG_DIR, mode='test')
    h14_test_loader = DataLoader(
        h14_test_dataset,
        batch_size=CLIPConfig_H14.BATCH_SIZE,
        shuffle=False,
        num_workers=CLIPConfig_H14.NUM_WORKERS,
        pin_memory=True
    )
    
    h14_sample_ids, h14_test_preds = predict_test_set(clip_h14_model, h14_test_loader, device, 'clip_h14')
    
    # Predict with CLIP-ViT-Large
    print("\nGenerating predictions with CLIP-ViT-Large...")
    large_test_dataset = CLIPDataset(test_df, processor_large, CLIPConfig_Large.TEST_IMG_DIR, mode='test')
    large_test_loader = DataLoader(
        large_test_dataset,
        batch_size=CLIPConfig_Large.BATCH_SIZE,
        shuffle=False,
        num_workers=CLIPConfig_Large.NUM_WORKERS,
        pin_memory=True
    )
    
    large_sample_ids, large_test_preds = predict_test_set(clip_large_model, large_test_loader, device, 'clip_large')
    
    # Predict with EfficientNet-BERT
    print("\nGenerating predictions with EfficientNet-BERT...")
    bert_test_dataset = BERTDataset(test_df, bert_tokenizer, BERTConfig.TEST_IMG_DIR, mode='test')
    bert_test_loader = DataLoader(
        bert_test_dataset,
        batch_size=BERTConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=BERTConfig.NUM_WORKERS,
        pin_memory=True
    )
    
    bert_sample_ids, bert_test_preds = predict_test_set(bert_model, bert_test_loader, device, 'bert')
    
    pred_time = time.time() - pred_start
    print(f"\n✓ Test predictions completed in {pred_time:.1f}s")
    
    # Verify sample_ids match
    assert h14_sample_ids == large_sample_ids == bert_sample_ids, "Sample IDs don't match!"
    
    # ==================== CREATE ENSEMBLE PREDICTIONS ====================
    
    print("\n" + "-"*70)
    print("CREATING ENSEMBLE PREDICTIONS")
    print("-"*70)
    
    # Apply best ensemble method
    if best_method == 'simple_average':
        final_preds = (h14_test_preds + large_test_preds + bert_test_preds) / 3
        print("\nUsing Simple Average ensemble")
        
    elif best_method == 'optimized_weighted':
        w1, w2, w3 = opt_weights
        final_preds = weighted_ensemble_3(h14_test_preds, large_test_preds, bert_test_preds, w1, w2, w3)
        print(f"\nUsing Optimized Weighted ensemble")
        print(f"  Weights: H14={w1:.3f}, Large={w2:.3f}, BERT={w3:.3f}")
        
    elif best_method == 'geometric_mean':
        final_preds = geometric_mean_ensemble_3(h14_test_preds, large_test_preds, bert_test_preds)
        print("\nUsing Geometric Mean ensemble")
        
    else:  # harmonic_mean
        final_preds = harmonic_mean_ensemble_3(h14_test_preds, large_test_preds, bert_test_preds)
        print("\nUsing Harmonic Mean ensemble")
    
    # ==================== CREATE SUBMISSION FILES ====================
    
    print("\n" + "-"*70)
    print("CREATING SUBMISSION FILES")
    print("-"*70)
    
    output_dir = "/home/kumaraa_iitp/hatter/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create main submission file (best ensemble)
    submission_df = pd.DataFrame({
        'sample_id': h14_sample_ids,
        'price': final_preds
    })
    
    main_submission_path = os.path.join(output_dir, f"triple_ensemble_{best_method}_submission.csv")
    submission_df.to_csv(main_submission_path, index=False)
    print(f"\n✓ Main submission saved: {main_submission_path}")
    print(f"  Method: {best_method}")
    print(f"  Expected SMAPE: ~{best_method_smape:.2f}%")
    
    # Create individual model submissions for comparison
    h14_submission = pd.DataFrame({'sample_id': h14_sample_ids, 'price': h14_test_preds})
    h14_path = os.path.join(output_dir, "clip_h14_test_predictions.csv")
    h14_submission.to_csv(h14_path, index=False)
    print(f"\n✓ CLIP-ViT-H-14 predictions saved: {h14_path}")
    
    large_submission = pd.DataFrame({'sample_id': large_sample_ids, 'price': large_test_preds})
    large_path = os.path.join(output_dir, "clip_large_test_predictions.csv")
    large_submission.to_csv(large_path, index=False)
    print(f"✓ CLIP-ViT-Large predictions saved: {large_path}")
    
    bert_submission = pd.DataFrame({'sample_id': bert_sample_ids, 'price': bert_test_preds})
    bert_path = os.path.join(output_dir, "efficientnet_bert_test_predictions.csv")
    bert_submission.to_csv(bert_path, index=False)
    print(f"✓ EfficientNet-BERT predictions saved: {bert_path}")
    
    # Create all ensemble variants
    print("\n" + "-"*50)
    print("Creating all ensemble variant submissions...")
    print("-"*50)
    
    # Simple average
    simple_test_preds = (h14_test_preds + large_test_preds + bert_test_preds) / 3
    simple_submission = pd.DataFrame({'sample_id': h14_sample_ids, 'price': simple_test_preds})
    simple_path = os.path.join(output_dir, "triple_ensemble_simple_average.csv")
    simple_submission.to_csv(simple_path, index=False)
    print(f"✓ Simple Average: {simple_path}")
    
    # Optimized weighted
    opt_test_preds = weighted_ensemble_3(h14_test_preds, large_test_preds, bert_test_preds, 
                                         opt_weights[0], opt_weights[1], opt_weights[2])
    opt_submission = pd.DataFrame({'sample_id': h14_sample_ids, 'price': opt_test_preds})
    opt_path = os.path.join(output_dir, "triple_ensemble_optimized_weighted.csv")
    opt_submission.to_csv(opt_path, index=False)
    print(f"✓ Optimized Weighted: {opt_path}")
    
    # Geometric mean
    geo_test_preds = geometric_mean_ensemble_3(h14_test_preds, large_test_preds, bert_test_preds)
    geo_submission = pd.DataFrame({'sample_id': h14_sample_ids, 'price': geo_test_preds})
    geo_path = os.path.join(output_dir, "triple_ensemble_geometric_mean.csv")
    geo_submission.to_csv(geo_path, index=False)
    print(f"✓ Geometric Mean: {geo_path}")
    
    # Harmonic mean
    harm_test_preds = harmonic_mean_ensemble_3(h14_test_preds, large_test_preds, bert_test_preds)
    harm_submission = pd.DataFrame({'sample_id': h14_sample_ids, 'price': harm_test_preds})
    harm_path = os.path.join(output_dir, "triple_ensemble_harmonic_mean.csv")
    harm_submission.to_csv(harm_path, index=False)
    print(f"✓ Harmonic Mean: {harm_path}")
    
    # ==================== SUMMARY STATISTICS ====================
    
    print("\n" + "="*70)
    print("PREDICTION STATISTICS")
    print("="*70)
    
    print(f"\nFinal Ensemble Predictions ({best_method}):")
    print(f"  Min price:    ${final_preds.min():.2f}")
    print(f"  Max price:    ${final_preds.max():.2f}")
    print(f"  Mean price:   ${final_preds.mean():.2f}")
    print(f"  Median price: ${np.median(final_preds):.2f}")
    print(f"  Std dev:      ${final_preds.std():.2f}")
    
    print("\nPrice distribution:")
    bins = [0, 10, 25, 50, 100, 250, 500, 1000, np.inf]
    labels = ['$0-10', '$10-25', '$25-50', '$50-100', '$100-250', '$250-500', '$500-1k', '$1k+']
    price_dist = pd.cut(final_preds, bins=bins, labels=labels)
    for label in labels:
        count = (price_dist == label).sum()
        pct = count / len(final_preds) * 100
        print(f"  {label:12s}: {count:5d} ({pct:5.1f}%)")
    
    # ==================== VALIDATION SUMMARY ====================
    
    print("\n" + "="*70)
    print("VALIDATION PERFORMANCE SUMMARY")
    print("="*70)
    
    print("\nIndividual Models:")
    print(f"  CLIP-ViT-H-14:      SMAPE = {h14_results['smape']:.2f}%")
    print(f"  CLIP-ViT-Large:     SMAPE = {large_results['smape']:.2f}%")
    print(f"  EfficientNet-BERT:  SMAPE = {bert_results['smape']:.2f}%")
    
    print("\nEnsemble Methods:")
    print(f"  Simple Average:     SMAPE = {simple_smape:.2f}%")
    print(f"  Optimized Weighted: SMAPE = {opt_weighted_smape:.2f}%")
    print(f"  Geometric Mean:     SMAPE = {geo_smape:.2f}%")
    print(f"  Harmonic Mean:      SMAPE = {harm_smape:.2f}%")
    
    print(f"\n🏆 BEST METHOD: {best_method.upper()}")
    print(f"   Validation SMAPE: {best_method_smape:.2f}%")
    
    improvement = min(h14_results['smape'], large_results['smape'], bert_results['smape']) - best_method_smape
    print(f"   Improvement over best single model: {improvement:.2f}%")
    
    # ==================== FINAL SUMMARY ====================
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    
    print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Model loading:     {load_time:.1f}s")
    print(f"  Validation eval:   {eval_time:.1f}s")
    print(f"  Test prediction:   {pred_time:.1f}s")
    
    print(f"\n✓ Main submission file: {main_submission_path}")
    print(f"✓ Total predictions: {len(final_preds)}")
    print(f"✓ Expected SMAPE: ~{best_method_smape:.2f}%")
    
    print("\n" + "="*70)
    print("All files saved successfully!")
    print("="*70 + "\n")
    
    return {
        'best_method': best_method,
        'best_smape': best_method_smape,
        'optimal_weights': opt_weights,
        'submission_path': main_submission_path,
        'predictions': final_preds,
        'sample_ids': h14_sample_ids
    }

if __name__ == "__main__":
    results = main()