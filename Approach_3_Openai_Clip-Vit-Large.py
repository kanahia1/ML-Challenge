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
    SUBMISSION_PATH = "/home/kumaraa_iitp/hatter/dataset/train.csv"
    MODEL_SAVE_PATH = "/home/kumaraa_iitp/hatter/output/best_clip_price_model.pth"
    
    CLIP_MODEL = "openai/clip-vit-large-patch14"
    MAX_TEXT_LENGTH = 77
    
    BATCH_SIZE = 40
    EPOCHS = 7  
    LEARNING_RATE_CLIP = 3e-6
    LEARNING_RATE_HEAD = 5e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.3
    
    VAL_SPLIT = 0.15
    WARMUP_EPOCHS = 3
    
    PATIENCE = 7
    
    USE_AMP = True
    
    SEED = 42
    NUM_WORKERS = 4

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.price_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
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

def main():
    print("="*60)
    print("CLIP-based Product Price Prediction - RESUME TRAINING")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print(f"\nLoading training data from {Config.TRAIN_CSV_PATH}")
    train_df = pd.read_csv(Config.TRAIN_CSV_PATH)
    print(f"Training samples: {len(train_df)}")

    train_data, val_data = train_test_split(
        train_df, 
        test_size=Config.VAL_SPLIT, 
        random_state=Config.SEED
    )
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")

    print(f"\nInitializing CLIP processor and model...")
    processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL)
    model = CLIPPricePredictor().to(device)

    print(f"\nLoading checkpoint from {Config.MODEL_SAVE_PATH}")
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_smape = checkpoint['smape']
    print(f"✓ Checkpoint loaded!")
    print(f"Previous best SMAPE: {best_smape:.2f}%")
    print(f"Resuming from epoch {start_epoch + 1}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e9:.2f}B")
    print(f"Trainable parameters: {trainable_params/1e9:.2f}B")

    print("\nCreating datasets...")
    train_dataset = ProductPriceDataset(train_data, processor, Config.TRAIN_IMG_DIR, mode='train')
    val_dataset = ProductPriceDataset(val_data, processor, Config.TRAIN_IMG_DIR, mode='train')

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    criterion = SMAPELoss()

    optimizer = torch.optim.AdamW([
        {'params': model.clip.parameters(), 'lr': Config.LEARNING_RATE_CLIP},
        {'params': model.price_head.parameters(), 'lr': Config.LEARNING_RATE_HEAD}
    ], weight_decay=Config.WEIGHT_DECAY)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        min_lr=1e-7
    )

    scaler = torch.cuda.amp.GradScaler() if Config.USE_AMP else None

    print(f"\nResuming training for {Config.EPOCHS} more epochs...")
    print(f"Total epochs will be: {start_epoch + Config.EPOCHS}")
    print("-"*60)

    patience_counter = 0

    for epoch in range(start_epoch, start_epoch + Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{start_epoch + Config.EPOCHS}")
        
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

if __name__ == "__main__":
    main()