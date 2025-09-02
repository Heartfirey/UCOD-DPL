from abc import ABCMeta, abstractmethod
from engine.config.config import CfgNode
from engine.utils.metrics.metric import calculate_cod_metrics, statistics
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import rich
from rich.progress import *
import math
from cv2 import connectedComponents,boundingRect
from torchvision import transforms
from PIL import Image, ImageChops
from data.utils.feature_extractor import backbone
from .utils import ProgressManager
import numpy as np
import os
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
class BaseLoop(metaclass=ABCMeta):
    def __init__(self, config: CfgNode, runner):
        self.cfg = config
        self._runner = runner
        self._dist_train = self.cfg.train_cfg.dist_train
    
    @property
    def runner(self):
        return self._runner

    @abstractmethod
    def run(self) -> Any:
        """Execute loop."""

        
class TrainLoop(BaseLoop):
    def __init__(self, config: CfgNode, runner):
        super().__init__(config, runner)

        # Dataset and model configuration
        self._mode = 'train'
        

        # Training configuration
        self._start_epoch = self.cfg.train_cfg.start_epoch
        self._max_epoch = self.cfg.train_cfg.max_epoch
        self.global_step = 0
        self._cur_epoch = 0
        self._start_finetune = self.cfg.train_cfg.start_finetune
        self.finetune = False
        
        # Loss and optimization
        self.criterion = nn.BCEWithLogitsLoss()
        self.dis_loss = nn.BCELoss()
        self.merge_alpha = self.cfg.train_cfg.merge_alpha
        self.ema_alpha = self.cfg.model_cfg.ema_weight

        # Validation and saving configuration
        self._setup_validation_config()
        self._setup_logging_config()

        # setup progress tracking
        self._setup_progress_manager()

        # Training state
        self.best_mae = 1000.0
        self.best_result = None

    def _setup_progress_manager(self) -> None:
        """Initialize and configure the progress manager."""
        self.progress_manager = ProgressManager(self.runner.accelerator)
        self.progress_manager.setup_progress()
        self.progress_manager.add_task("Train Iteration", total=len(self.runner.train_dataloader))
        self.progress_manager.add_task("Validation Iteration", total=len(self.runner.val_dataloader))
        self.progress_manager.add_task("Discriminator Iteration", total=len(self.runner.train_dataloader))
        self.progress_manager.add_task("Train Epoch", total=self.cfg.train_cfg.max_epoch)

    def _setup_validation_config(self) -> None:
        """Setup validation-related configuration parameters."""
        self.enable_val = self.cfg.val_cfg.enable_val
        self.val_interval = self.cfg.val_cfg.val_interval
        self.dis_intertrain = self.cfg.train_cfg.dis_intertrain
        self.val_start = (self._max_epoch + self.cfg.val_cfg.start_val) \
                            if self.cfg.val_cfg.start_val < 0 else self.cfg.val_cfg.start_val
        
        self.save_start = (self._max_epoch + self.cfg.train_cfg.save_cfg.start_save) \
                            if self.cfg.train_cfg.save_cfg.start_save < 0 else self.cfg.train_cfg.save_cfg.start_save
        self.save_interval = self.cfg.train_cfg.save_cfg.save_interval

    def _setup_logging_config(self) -> None:
        """Setup logging-related configuration parameters."""
        self.log_interval = self.cfg.log_cfg.log_interval

    def run(self):
        self.runner.logger.log(self.cfg)

        with self.progress_manager:
            self.progress_manager.start_task('Train Epoch')

            while self._cur_epoch < self._max_epoch:
                if self.decide_to_finetune():
                    self.runner.start_finetune()
                    self.global_step=0

                if self.decide_to_train_dis():
                    self.Discriminator_train()

                self.run_epoch()
                self._cur_epoch += 1

                if self.decide_to_save():
                    self.runner.save_checkpoint(self._cur_epoch)

                if self.decide_to_val():
                    result = self.runner.launch_val_look_twice()
                    self._update_best_result(result)

                self.progress_manager.update_task('Train Epoch')

    def _update_best_result(self, result: Dict[str, float]) -> None:
        """
        Update best validation result if current result is better.
        
        Args:
            result: Dictionary containing validation metrics
        """
        mae = result["MAE"]
        if mae < self.best_mae:
            self.best_mae = mae
            self.best_result = result
            result_table = {key: [round(result[key], 4)] for key in result.keys()}
            self.runner.logger.log("best result:")
            self.runner.logger.log_table(result_table)

    def run_epoch(self):
        self.runner.model.train()
        self.progress_manager.start_task("Train Iteration")

        for batch_data in self.runner.train_dataloader:
            loss = self._process_batch(batch_data)
            if self._cur_epoch % self.log_interval == 0:
                self.runner.logger.log(f"iter{self.global_step}:loss:{loss:.4f}")
            self.global_step += 1
            self.progress_manager.update_task("Train Iteration")
        
        self.progress_manager.reset_task("Train Iteration")

    def _process_batch(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        pseudo_labels, _, features, _ = batch_data.values()
        self.runner.optimizer.zero_grad()
        
        h = w = self.cfg.model_cfg.feature_size
        features = F.interpolate(features, size=(h,w), mode='bilinear')     
        pseudo_labels = F.interpolate(pseudo_labels, size=(h,w), mode='bilinear').float()

        with torch.no_grad():
            preds_ema = self.runner.model(features, ema=True)
        preds, preds_rev, extra_loss = self.runner.model(features)

        pseudo_labels, dis_loss = self.merge_pseudo_label(pseudo_labels, preds_ema, preds, features)
        flat_pseudo_labels = pseudo_labels.permute(0, 2, 3, 1).reshape(-1, 1)
        flat_preds = preds.permute(0, 2, 3, 1).reshape(-1, 1)
        flat_preds_rev = preds_rev.permute(0, 2, 3, 1).reshape(-1, 1)
        loss = self.criterion(
            flat_preds, flat_pseudo_labels
        )
        if dis_loss != None and not self.finetune:
            loss -= dis_loss
            self.runner.logger.log('train/dis_loss:{:.4f}'.format(dis_loss))
        
        loss += self.criterion(flat_preds_rev, (1 - flat_pseudo_labels))
        if extra_loss is not None:
            loss += extra_loss

        self.runner.logger.log("iter{}:loss:{:.4f}".format(self.global_step, loss))

        self.runner.accelerator.backward(loss)
        self.runner.optimizer.step()
        self.runner.lr_scheduler.step()
        
        self.update_ema_decoder()
        self.global_step+=1
                          
        return loss

    def update_ema_decoder(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_alpha)
        for ema_param, param in zip(self.runner.model.decoder_ema.parameters(), self.runner.model.decoder.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        for ema_buffer, buffer in zip(self.runner.model.decoder_ema.buffers(), self.runner.model.decoder.buffers()):
            ema_buffer.data.copy_(buffer.data)

    def decide_to_train_dis(self):
        if self.cfg.train_cfg.merge_method != 'dis':
            return False
        if self._cur_epoch % self.dis_intertrain == 0 and not self.finetune:
            return True
        return False

    def decide_to_finetune(self):
        if self._cur_epoch == self._max_epoch + self._start_finetune:
            self.finetune = True
            return True
        return False

    def decide_to_save(self) -> bool:
        return (self._cur_epoch >= self.save_start and 
                self._cur_epoch % self.save_interval == 0)

    def decide_to_val(self) -> bool:
        return (self.enable_val and 
                self._cur_epoch >= self.val_start and 
                self._cur_epoch % self.val_interval == 0)

    def Discriminator_train(self):
        for param in self.runner.discriminator.parameters():
            param.requires_grad  = True
        for param in self.runner.model.decoder.parameters():
            param.requires_grad = False
        for _ in range (self.cfg.train_cfg.dis_epoch):
            self.Discriminator_epoch()
        for param in self.runner.discriminator.parameters():
            param.requires_grad  = False
        for name, param in self.runner.model.decoder.named_parameters():
            if 'decoder_ema' not in name:
                param.requires_grad = True



    def Discriminator_epoch(self):
        self.progress_manager.start_task("Discriminator Iteration")
        for batch in self.runner.train_dataloader:
            self.runner.dis_optimizer.zero_grad()
            pseudo_labels, _, features, _ = batch.values()
            h = w = self.cfg.model_cfg.feature_size
            features = F.interpolate(features, size=(h,w), mode='bilinear')    
                
            with torch.no_grad():
                preds, _, _ = self.runner.model(features)
                preds = (torch.nn.functional.sigmoid(preds.detach())>0.5).float()
            pseudo_labels = (F.interpolate(pseudo_labels, size=(preds.shape[2],preds.shape[3]), mode='bilinear')>0.5).float()
            batch_size = preds.shape[0]
            label = torch.cat((torch.zeros(batch_size),torch.ones(batch_size)), dim=-1).unsqueeze(-1).to('cuda')
            probs_pseudo = self.runner.discriminator(pseudo_labels, features)
            probs_student = self.runner.discriminator(preds, features)
            probs = torch.cat((probs_student, probs_pseudo), dim=0)
            loss = self.dis_loss(probs, label)

            self.runner.logger.log("dis:loss:{:.4f}".format(loss))
            self.runner.accelerator.backward(loss)
            self.runner.dis_optimizer.step()
            self.runner.dis_lr_scheduler.step()

            self.progress_manager.update_task("Discriminator Iteration")
        self.progress_manager.reset_task("Discriminator Iteration")

    def merge_pseudo_label(self, pseudo_labels, p_teachers, p_students, features):
        p_teachers = (p_teachers.sigmoid()>0.5).float()
        p_students = (p_students.sigmoid()>0.5).float()
        p_s = self.runner.discriminator(p_students, features)
        p_p = self.runner.discriminator((pseudo_labels>0.5).float(), features)

        self.runner.logger.log('pl:{:.4f}'.format(p_p.mean()))
        self.runner.logger.log('ps:{:.4f}'.format(p_s.mean()))

        weight = 0.5*(1 + torch.cos( torch.abs(p_s - p_p) * torch.pi )) + self._cur_epoch / (self._max_epoch + self._start_finetune)
        weight = torch.clamp(weight, 0, 1)
        weight = weight.unsqueeze(-1).unsqueeze(-1)
        self.runner.logger.log("merge_label_weight:{:.2f}".format(weight.mean()))
        target = torch.zeros(p_students.shape[0]).unsqueeze(1).to('cuda')
        loss = self.dis_loss(p_s, target)
        return pseudo_labels * (1 - weight) + p_teachers * weight, loss

    

class ValLoop_Look_Twice(BaseLoop):
    def __init__(self, config: CfgNode, runner):
        super().__init__(config, runner)
        self._mode = 'val'
        self._runner = runner
        self.img_size = self.cfg.dataset_cfg.valset_cfg.image_size
        self.transform_image = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.to_PIL = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.feature_extractor = backbone(self.cfg.dataset_cfg.feature_extractor_cfg)
        self.feature_extractor = self.runner.accelerator.prepare(self.feature_extractor)
        self.key = []
        
        self.progress_manager = ProgressManager(self.runner.accelerator)
        self.progress_manager.setup_progress()
        self.progress_manager.add_task("Validation Iteration", total=len(self.runner.val_dataloader))
        
    def run(self):        
        statistics_val = statistics()
        self.runner.model.eval()
        self.progress_manager.start_task("Validation Iteration")
        for batch in tqdm(self.runner.val_dataloader):
            _, label_tensor, features, img_path = batch.values()
            bs = label_tensor.shape[0]
            h = w = self.cfg.model_cfg.feature_size
            features = F.interpolate(features, size=(h,w), mode='bilinear')       

            with torch.no_grad():
                preds, _, _ = self.runner.model(features)
                # gather all preds and labels
                all_preds, all_labels = self.runner.accelerator.gather_for_metrics((preds, label_tensor))
                preds_up, bboxes = self.process_preds(all_preds, all_labels)
                if bboxes != None and self.cfg.val_cfg.look_twice:
                    new_mask = self.look_twice(img_path[0], bboxes, preds_up).to('cuda')
                    preds_up = new_mask
                preds_up = F.interpolate(preds_up.unsqueeze(0), size=(all_labels.shape[-2],all_labels.shape[-1]), mode='bilinear').squeeze(0)
                statistics_val.step(all_labels, preds_up>0.5)
                save_tensor_binary_mask_as_image(preds_up[0]>0.5, os.path.join(os.path.join(self.cfg.log_cfg.log_path, 'preds', self.cfg.dataset_cfg.valset_cfg.DATASET),os.path.basename(img_path[0])))
            self.progress_manager.update_task("Validation Iteration")
        # calculate_cod_metrics()
        result = statistics_val.get_result()
        result_table = {key:[round(result[key], 4)] for key in result.keys()}
        self.runner.logger.log_table(result_table)
        self.progress_manager.reset_task("Validation Iteration")
        return result
    
    def look_twice(self, path, bboxes, old_mask):
        ih, iw = self.img_size
        img = Image.open(path)

        array = (old_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        new_mask = Image.fromarray(array)

        for bbox in bboxes:
            new_bbox = self.resize_bbox(bbox, iw, ih, img.size[0], img.size[1])
            x, y, w, h = new_bbox
            left = x
            top = y
            right = x + w
            bottom = y + h
            cropped_img = img.crop((left, top, right, bottom))
            img_tensor = self.transform_image(cropped_img).to('cuda').unsqueeze(0)
            _, features = self.feature_extractor(img_tensor)
            with torch.no_grad():
                preds, _, _ = self.runner.model(features)
                if len(preds.shape) == 4:
                    preds = preds.squeeze(0)
            pred = (torch.nn.functional.sigmoid(preds.detach()) > 0.5).squeeze(0).float()
            pred_PIL = self.to_PIL(pred)
            pred_PIL = pred_PIL.resize((bbox[-2], bbox[-1]))
            new_mask.paste(pred_PIL, (bbox[0], bbox[1]))
        return self.to_tensor(new_mask) 

    def process_preds(self, preds, label_tensor):
        h ,w = self.img_size
        preds_up = F.interpolate(
            preds, size=(h,w), mode="bilinear", align_corners=False
        )[..., :h, :w]
        preds_up = (
            (torch.nn.functional.sigmoid(preds_up.detach()) > 0.5).squeeze(0).float()
        )
        np_preds_up = preds_up.to('cpu').numpy()
        np_preds_up = (np_preds_up * 255).astype(np.uint8)
        if len(np_preds_up.shape) == 3:
            np_preds_up = np_preds_up.squeeze(0)
        num_labels, labels = connectedComponents(np_preds_up, connectivity=8)
        
        p = [(labels==i).sum()/(h*w) for i in range(1,num_labels)]
        if len(p) ==0:
            return preds_up, [[129, 129, 259, 259]]
        p_max = max(p)
        if p_max < self.cfg.val_cfg.look_twice_th:
            bboxes = []
            for i in range(1, num_labels):
                if p[i-1]>0.01:
                    binary_mask = (labels == i).astype(np.uint8)
                    bbox = boundingRect(binary_mask)
                    bboxes.append(
                        self.expand_bbox(binary_mask, bbox, h, w, expand_type = self.cfg.val_cfg.expand_type)
                    )
            bboxes = sorted(bboxes, key=lambda bbox: -1*bbox[2] * bbox[3])
            return preds_up, bboxes
        else:
            return preds_up, None

    
    def resize_bbox(self, bbox, original_width, original_height, new_width, new_height):
        x, y, w, h = bbox
        width_scale = new_width / original_width
        height_scale = new_height / original_height
        
        new_x = int(x * width_scale)
        new_y = int(y * height_scale)
        new_w = int(w * width_scale)
        new_h = int(h * height_scale)
    
        return [new_x, new_y, new_w, new_h]

    def expand_bbox(self, mask, bbox, img_width, img_height, expand_type='const' ,scale=1.3):
        x, y, w, h = bbox

        if expand_type == 'dynamic':
            fr = mask[y:y+h, x:x+w].sum() / (h * w)
            br = (h * y) / (mask.shape[-2] * mask.shape[-1])
            scale = math.sqrt(1 - br/fr +1)
            
        new_w = w * scale
        new_h = h * scale
        new_x = x - (new_w - w) / 2
        new_y = y - (new_h - h) / 2
        new_x = max(0, new_x)
        if new_x + new_w > img_width:
            new_x = img_width - new_w
        new_y = max(0, new_y)
        if new_y + new_h > img_height:
            new_y = img_height - new_h
        return [int(new_x), int(new_y), int(new_w), int(new_h)]


def save_tensor_binary_mask_as_image(binary_mask, save_path):
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask.squeeze(0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    binary_mask_np = (binary_mask.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(binary_mask_np, mode='L')
    img.save(save_path.replace('.jpg','.png'))