import torch
import logging
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from utils.utils import labels2Dto3D_flattened, flattenDetection, precisionRecall_torch
from models.SuperPointNet_gauss2 import SuperPointNet_gauss2
from utils.loader import dataLoader

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse
from utils.utils import descriptor_loss
from omegaconf import OmegaConf


class Train_model_frontend:
    def __init__(
        self, 
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,
        save_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        max_epochs: int = 200,
        early_stopping_patience: int = 10,
        ):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.critetion = nn.CrossEntropyLoss()
        
        self.lambda_det = config['model'].get('lambda_det', 1.0)
        self.lambda_desc = config['model'].get('lambda_desc', 1.0)
        self.detection_threshold = config['model'].get('detection_threshold', 0.015)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir)

        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Инициализация логгера
        self.setup_logging()
        
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def compute_loss(self, outputs, labels_2D, warped_labels, mask, mat_H):
        labels3D = labels2Dto3D_flattened(labels_2D, cell_size=8).to(self.device)
       
        loss_det = nn.functional.cross_entropy(
            outputs[0],
            labels3D,
            reduction='none'
        )
        
        mask = mask.squeeze(1)
        
        mask_downsampled = F.avg_pool2d(
            mask.unsqueeze(1).float(),  # [16, 1, 240, 320]
            kernel_size=8, 
            stride=8
        ).squeeze(1)
        
        mask_binary = (mask_downsampled > 0.5).float()
        valid_pixels = mask_binary.sum()
        
        loss_det = (loss_det * mask_binary).sum() / (valid_pixels + 1e-10)
    
        if self.config['model']['dense_loss']['enable']:
            loss_desc = descriptor_loss(
                outputs[1], # desc
                outputs[3], # desc_warp
                mat_H,
                mask_valid=mask,
                device=self.device,
                **self.config['model']['dense_loss']['params']
            )
        else:
            loss_desc = batch_descriptor_loss_sparse(
                outputs[1],  # desc
                outputs[3],  # desc_warp
                mat_H,
                mask_valid=mask,
                device=self.device,
                **self.config['model']['sparse_loss']['params']
            )
            

        total_loss = self.lambda_det * loss_det + self.lambda_desc * loss_desc[0]
        
        return {
            'total': total_loss,
            'det': loss_det,
            'desc': loss_desc
        }
        
    def train_epoch(self) -> Dict[str, float]:
        '''Обучение одной эпохи'''
        self.model.train()
        
        total_loss = 0
        total_det_loss = 0
        total_desc_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} Train")
        
        for batch_idx, batch in enumerate(pbar):
            
            img = batch[0].to(self.device)
            img = img.permute(0, 3, 1, 2)
            labels_2D = batch[1].to(self.device)
            labels_2D = labels_2D.unsqueeze(1)
            
            valid_mask = batch[2].to(self.device)
            warped_img = batch[3].to(self.device)
            warped_labels_res = batch[4].to(self.device)
            homography = batch[5].to(self.device)
                   
            self.optimizer.zero_grad()
            
            forward_res = self.model(img)
            semi, desc = forward_res['semi'], forward_res['desc']
            
            warped_res = self.model(warped_img)
            semi_warp, desc_warp = warped_res['semi'], warped_res['desc']

            loss_dict = self.compute_loss(
                [semi, desc, semi_warp, desc_warp],
                labels_2D,
                warped_labels_res,
                valid_mask,
                homography
            )
            
            loss = loss_dict['total']
            
            loss.backward()
            
            self.optimizer.step()
            

            total_loss += loss.item()
            total_det_loss += loss_dict['det'].item()
            total_desc_loss += loss_dict['desc'][0].item()
            
            avg_loss = total_loss / (batch_idx + 1)
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'det': f'{loss_dict["det"].item():.4f}',
                'desc': f'{loss_dict["desc"][0].item():.4f}'
            })
            
            if batch_idx % 10 == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss_batch', loss.item(), step)
                self.writer.add_scalar('Train/DetLoss_batch', loss_dict['det'].item(), step)
                self.writer.add_scalar('Train/DescLoss_batch', loss_dict['desc'][0].item(), step)
        
        epoch_metrics = {
            'loss': total_loss / len(self.train_loader),
            'det_loss': total_det_loss / len(self.train_loader),
            'desc_loss': total_desc_loss / len(self.train_loader)
        }
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0
        total_det_loss = 0
        total_desc_loss = 0
        
        # Для метрик детекции
        total_precision = 0
        total_recall = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} Train")
            
            for batch_idx, batch in enumerate(pbar):
                img = batch[0].to(self.device)
                labels_2D = batch[1].to(self.device)
                valid_mask = batch[2].to(self.device)
                warped_img = batch[3].to(self.device)
                warped_labels_res = batch[4].to(self.device)
                homography = batch[5].to(self.device)

                batch_size = img.shape[0]
                semi, desc = self.model(img)
                semi_warp, desc_warp = self.model(warped_img)
                
                loss_dict = self.compute_loss(
                    [semi, desc, semi_warp, desc_warp],
                    labels_2D,
                    warped_labels_res,
                    valid_mask,
                    homography
                )
                
                loss = loss_dict['total']
                
                batch_precision = []
                batch_recall = []
                
                for i in range(batch_size):
                    heatmap = flattenDetection(semi[i:i+1]).squeeze()
                    
                     # Бинаризация по порогу
                    heatmap_binary = (heatmap > self.detection_threshold).float()
                    
                    # Вычисляем precision и recall
                    metrics = precisionRecall_torch(
                        heatmap_binary.cpu(),
                        labels_2D[i].cpu()
                    )
                    batch_precision.append(metrics['precision'])
                    batch_recall.append(metrics['recall'])
                    
                total_loss += loss.item()
                total_det_loss += loss_dict['det'].item()
                total_desc_loss += loss_dict['desc'].item()
                total_precision += np.mean(batch_precision)
                total_recall += np.mean(batch_recall)
                
                avg_loss = total_loss / (batch_idx + 1)
                avg_precision = total_precision / (batch_idx + 1)
                avg_recall = total_recall / (batch_idx + 1)
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'prec': f'{avg_precision:.2%}',
                    'rec': f'{avg_recall:.2%}'
                })
                
                
            val_metrics = {
            'loss': total_loss / len(self.val_loader),
            'det_loss': total_det_loss / len(self.val_loader),
            'desc_loss': total_desc_loss / len(self.val_loader),
            'precision': total_precision / len(self.val_loader),
            'recall': total_recall / len(self.val_loader),
            'f1': 2 * (total_precision * total_recall) / 
                  (total_precision + total_recall + 1e-10) / len(self.val_loader)
            }
        
            return val_metrics
        
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'config': self.config,
        }
        
        torch.save(checkpoint, self.save_dir / filename)
        
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            self.logger.info(f"Saved best model to {self.save_dir / 'best_model.pth'}")
            
            
    def load_checkpoint(self, checkpoint_path: str):
        """Загрузка чекпоинта"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")
        
    def log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Логирование метрик в TensorBoard и консоль"""
        # TensorBoard
        self.writer.add_scalar('Loss/Train', train_metrics['loss'], self.current_epoch)
        self.writer.add_scalar('Loss/Val', val_metrics['loss'], self.current_epoch)
        self.writer.add_scalar('Loss/Det_Train', train_metrics['det_loss'], self.current_epoch)
        self.writer.add_scalar('Loss/Det_Val', val_metrics['det_loss'], self.current_epoch)
        self.writer.add_scalar('Loss/Desc_Train', train_metrics['desc_loss'], self.current_epoch)
        self.writer.add_scalar('Loss/Desc_Val', val_metrics['desc_loss'], self.current_epoch)
        
        # Метрики детекции
        self.writer.add_scalar('Metrics/Precision', val_metrics['precision'], self.current_epoch)
        self.writer.add_scalar('Metrics/Recall', val_metrics['recall'], self.current_epoch)
        self.writer.add_scalar('Metrics/F1', val_metrics['f1'], self.current_epoch)
        
        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('LR', current_lr, self.current_epoch)
        
        # Вывод в консоль
        self.logger.info(
            f"Epoch {self.current_epoch:03d} | "
            f"Train Loss: {train_metrics['loss']:.4f} (D:{train_metrics['det_loss']:.4f}, R:{train_metrics['desc_loss']:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} (D:{val_metrics['det_loss']:.4f}, R:{val_metrics['desc_loss']:.4f}) | "
            f"Prec: {val_metrics['precision']:.2%} | Rec: {val_metrics['recall']:.2%} | F1: {val_metrics['f1']:.3f} | "
            f"LR: {current_lr:.2e}"
        )
        
    def train(self):
        '''Основной цикл обучения'''
        self.logger.info('SuperPoint training started')
        self.logger.info("=" * 50)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Lambda det: {self.lambda_det}, Lambda desc: {self.lambda_desc}")
        self.logger.info(f"Detection threshold: {self.detection_threshold}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.max_epochs): 
            train_metrics = self.train_epoch()
            
            val_metrics = self.validate()
            
            self.log_metrics(train_metrics, val_metrics)
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pth', is_best=True)
                self.patience_counter = 0
                self.logger.info(f"New best model! Val loss: {val_metrics['loss']:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f'Stop triggered after {epoch} epochs')
                    break
                
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
                
        self.save_checkpoint('final_model.pth')

        self.writer.close()
        training_time = time.time() - start_time
        
        self.logger.info("=" * 50)
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Final model saved to: {self.save_dir / 'final_model.pth'}")
        
        
if __name__ == '__main__':
    cfg = OmegaConf.load("params.yaml")
    
    model = SuperPointNet_gauss2()
    data = dataLoader(cfg['prepare_synthetic_dataset'])
    
    train_loader, val_loader = data['train_loader'], data['val_loader']
    
    trainer = Train_model_frontend(
        model,
        train_loader, 
        val_loader,
        cfg,
        device='cuda',
        learning_rate=1e-3,
        save_dir='checkpoints/my_model',
        max_epochs=200,
        early_stopping_patience=20
    )
    
    trainer.train()
    