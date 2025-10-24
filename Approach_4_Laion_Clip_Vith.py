import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class Config:
    TRAIN_CSV_PATH = "/home/kumaraa_iitp/hatter/dataset/train.csv"
    TEST_CSV_PATH = "/home/kumaraa_iitp/hatter/dataset/test.csv"
    TRAIN_IMG_DIR = "/home/kumaraa_iitp/kanahia/src/complete_images/train_images"
    TEST_IMG_DIR = "/home/kumaraa_iitp/kanahia/src/complete_images/test_images"
    SUBMISSION_PATH = "/home/kumaraa_iitp/hatter/dataset/sub.csv"
    MODEL_SAVE_PATH = "/home/kumaraa_iitp/hatter/output/best_clip_vith_model.pth"
    
    CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    MAX_TEXT_LENGTH = 77
    
    BATCH_SIZE = 96  
    EPOCHS = 15  
    LEARNING_RATE_CLIP = 2e-6 
    LEARNING_RATE_HEAD = 1.5e-4  
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.35
    
    VAL_SPLIT = 0.15
    WARMUP_EPOCHS = 2 
    
    PATIENCE = 5
    
    USE_AMP = True
    
    SEED = 42
    NUM_WORKERS = 8

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.benchmark = True  

set_seed(Config.SEED)

class ProductPriceDataset(Dataset):
    def __init__(self, df, processor, image_dir, mode='train', testing=False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.image_dir = image_dir
        self.mode = mode
        self.testing = testing
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        sample_id = row['sample_id']
        
        try:
            if self.testing:
                image_filename = f"test_{sample_id}.jpg"
            else:
                image_filename = f"train_{sample_id}.jpg"
            
            image_path = os.path.join(self.image_dir, image_filename)
            
            if not os.path.exists(image_path):
                for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                    if self.testing:
                        alt_path = os.path.join(self.image_dir, f"test_{sample_id}{ext}")
                    else:
                        alt_path = os.path.join(self.image_dir, f"train_{sample_id}{ext}")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
            
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            print(f"Warning: Could not load image for {sample_id}, using blank image")
        
        text = str(row['catalog_content'])
        if pd.isna(text) or text == 'nan':
            text = "No description available"
        
        try:
            inputs = self.processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=Config.MAX_TEXT_LENGTH
            )
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            inputs = self.processor(
                text=["No description"],
                images=Image.new('RGB', (224, 224)),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=Config.MAX_TEXT_LENGTH
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

class CLIPPricePredictor(nn.Module):
    def __init__(self, clip_model_name=Config.CLIP_MODEL, dropout=Config.DROPOUT):
        super().__init__()
        
        print(f"Loading CLIP model: {clip_model_name}")
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        embed_dim = self.clip.projection_dim 
        print(f"Embedding dimension: {embed_dim}")
        
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
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.price_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
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
        
        log_price = self.price_head(combined)
        return log_price.squeeze(-1)

class SMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        pred_original = torch.expm1(pred)
        target_original = torch.expm1(target)
        
        numerator = torch.abs(pred_original - target_original)
        denominator = (torch.abs(target_original) + torch.abs(pred_original)) / 2 + self.epsilon
        smape = torch.mean(numerator / denominator)
        
        return smape

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, epoch=0, warmup_epochs=Config.WARMUP_EPOCHS):
    model.train()
    
    if epoch < warmup_epochs:
        for param in model.clip.parameters():
            param.requires_grad = False
    else:
        for param in model.clip.parameters():
            param.requires_grad = True
    
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    
    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        log_prices = batch['price'].to(device)
        
        optimizer.zero_grad()
        
        if scaler and Config.USE_AMP:
            with torch.cuda.amp.autocast():
                predictions = model(pixel_values, input_ids, attention_mask)
                loss = criterion(predictions, log_prices)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(pixel_values, input_ids, attention_mask)
            loss = criterion(predictions, log_prices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def validate(model, dataloader, device):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            log_prices = batch['price'].to(device)
            
            with torch.cuda.amp.autocast():
                predictions = model(pixel_values, input_ids, attention_mask)
            
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(log_prices.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    preds_original = np.expm1(all_preds)
    targets_original = np.expm1(all_targets)
    
    preds_original = np.maximum(preds_original, 0.01)
    
    smape = np.mean(np.abs(preds_original - targets_original) / 
                    ((np.abs(targets_original) + np.abs(preds_original)) / 2 + 1e-8)) * 100
    
    mae = np.mean(np.abs(preds_original - targets_original))
    rmse = np.sqrt(np.mean((preds_original - targets_original) ** 2))
    
    return smape, mae, rmse

def predict(model, dataloader, device):
    model.eval()
    
    all_sample_ids = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sample_ids = batch['sample_id']
            
            with torch.cuda.amp.autocast():
                log_predictions = model(pixel_values, input_ids, attention_mask)
            
            predictions = torch.expm1(log_predictions)
            
            predictions = torch.clamp(predictions, min=0.01)
            
            all_sample_ids.extend(sample_ids)
            all_predictions.extend(predictions.cpu().numpy())
    
    return all_sample_ids, all_predictions


def main():
    print("="*60)
    print("CLIP-based Product Price Prediction (ViT-H-14)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"\nLoading training data from {Config.TRAIN_CSV_PATH}")
    train_df = pd.read_csv(Config.TRAIN_CSV_PATH)
    print(f"Training samples: {len(train_df)}")

    print("\nPrice Statistics:")
    print(f"Min: ${train_df['price'].min():.2f}")
    print(f"Max: ${train_df['price'].max():.2f}")
    print(f"Mean: ${train_df['price'].mean():.2f}")
    print(f"Median: ${train_df['price'].median():.2f}")
    print(f"Skewness: {train_df['price'].skew():.2f}")

    train_data, val_data = train_test_split(
        train_df, 
        test_size=Config.VAL_SPLIT, 
        random_state=Config.SEED
    )
    print(f"\nTrain set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")

    print(f"\nInitializing CLIP processor and model...")
    processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL)
    model = CLIPPricePredictor().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e9:.2f}B")
    print(f"Trainable parameters: {trainable_params/1e9:.2f}B")
    print(f"Model: ViT-H-14 (986M parameters) with enhanced regression head")

    print("\nCreating datasets...")
    train_dataset = ProductPriceDataset(train_data, processor, Config.TRAIN_IMG_DIR, mode='train')
    val_dataset = ProductPriceDataset(val_data, processor, Config.TRAIN_IMG_DIR, mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )

    criterion = SMAPELoss()

    optimizer = torch.optim.AdamW([
        {'params': model.clip.parameters(), 'lr': Config.LEARNING_RATE_CLIP},
        {'params': model.price_head.parameters(), 'lr': Config.LEARNING_RATE_HEAD}
    ], weight_decay=Config.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2,
        min_lr=1e-7
    )

    scaler = torch.cuda.amp.GradScaler() if Config.USE_AMP else None

    print(f"\nStarting training for {Config.EPOCHS} epochs...")
    print(f"Warmup epochs (head only): {Config.WARMUP_EPOCHS}")
    print(f"Batch size: {Config.BATCH_SIZE} (optimized for A100)")
    print("-"*60)

    best_smape = float('inf')
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch
        )
        
        val_smape, val_mae, val_rmse = validate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val SMAPE: {val_smape:.2f}%")
        print(f"Val MAE: ${val_mae:.2f}")
        print(f"Val RMSE: ${val_rmse:.2f}")
        
        scheduler.step(val_smape)
        
        if val_smape < best_smape:
            best_smape = val_smape
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'smape': val_smape
            }, Config.MODEL_SAVE_PATH)
            print(f"✓ Model saved! Best SMAPE: {best_smape:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{Config.PATIENCE}")
        
        if patience_counter >= Config.PATIENCE:
            print("\nEarly stopping triggered!")
            break

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best Validation SMAPE: {best_smape:.2f}%")
    print("="*60)

    print("\nLoading best model for test predictions...")
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nLoading test data from {Config.TEST_CSV_PATH}")
    test_df = pd.read_csv(Config.TEST_CSV_PATH)
    print(f"Test samples: {len(test_df)}")

    test_dataset = ProductPriceDataset(test_df, processor, Config.TEST_IMG_DIR, mode='test', testing=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    print("\nGenerating predictions...")
    sample_ids, predictions = predict(model, test_loader, device)

    submission_df = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })

    if len(submission_df) != len(test_df):
        print(f"Warning: Prediction count mismatch!")
        print(f"Expected: {len(test_df)}, Got: {len(submission_df)}")

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"\n✓ Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Sample predictions:")
    print(submission_df.head(10))

    print("\n" + "="*60)
    print("All done! Good luck with the competition!")
    print("="*60)

if __name__ == "__main__":
    main()