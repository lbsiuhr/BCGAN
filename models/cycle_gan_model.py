import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import os
import cv2

class CycleGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Configure BCGAN options: Cycle consistency and physics-informed scanning module.
        """
        parser.set_defaults(no_dropout=True)
        parser.set_defaults(no_flip=True)

        if is_train:
            # --- 1. Base weights ---
            parser.add_argument('--lambda_A', type=float, default=20.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=20.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the loss')

            # --- 2. Physics-informed scanning module ---
            # Principle: Generated cells -> Simulated motion blur along flow direction -> Matched with GT mask via Dice Loss
            parser.add_argument('--lambda_physics', type=float, default=1.5,
                                help='maximum weight for physics-informed consistency loss')

            parser.add_argument('--physics_kernels', type=str, default='20,15,23,27',
                                help='comma-separated kernel sizes simulating motion blur elongation along the X-axis')

            parser.add_argument('--mask_expand_radius', type=int, default=2,
                                help='tolerance dilation radius for fake mask to avoid edge sensitivity')
            
            parser.add_argument('--weight_dice', type=float, default=1.0,
                                help='weight for Dice loss between blurred fake mask and GT mask')

            parser.add_argument('--weight_proj', type=float, default=1.5,
                                help='weight for direct Y-axis projection loss enforcing vertical boundary alignment')

            # --- 2b. Differentiable soft mask extraction ---
            parser.add_argument('--mask_denoise_k', type=int, default=7,
                                help='Gaussian pre-smoothing kernel size before adaptive thresholding')
            parser.add_argument('--adaptive_block_size', type=int, default=35,
                                help='local neighborhood radius for adaptive thresholding (must be odd)')
            parser.add_argument('--adaptive_thresh_c', type=float, default=0.04,
                                help='offset constant subtracted from global mean for threshold calculation')
            parser.add_argument('--mask_sigmoid_gain', type=float, default=50.0,
                                help='gain factor controlling sigmoid sharpness for soft binarization')

            # --- 3. Dynamic weight schedule ---
            # Phases: [0] -> Warmup -> [Max] -> Sustain -> [Max] -> Decay -> [0]
            parser.add_argument('--phys_warmup_start', type=int, default=0, help='epoch at which physics loss begins')
            parser.add_argument('--phys_warmup_end', type=int, default=3, help='epoch at which physics loss reaches max weight')
            parser.add_argument('--phys_decay_start', type=int, default=100, help='epoch at which physics loss begins linear decay')
            parser.add_argument('--phys_decay_end', type=int, default=200, help='epoch at which physics loss completely reaches zero')
            
            parser.add_argument('--phys_schedule_mode', type=str, default='absolute', choices=['absolute', 'relative'],
                                help='scheduling mode: absolute (by global epoch) or relative (by elapsed epochs from epoch_count)')

            # --- 4. Dataset paths ---
            parser.add_argument('--mask_dataroot', type=str,
                                default='./datasets/jurkat_mini_sample/masks_visible',
                                help='directory containing ground truth mask images')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        if self.isTrain:
            self.kernel_list = [int(k) for k in opt.physics_kernels.split(',')]
            print("------------ BCGAN Configuration ------------")
            print(f" Physics scanning: ENABLED (Max Lambda = {opt.lambda_physics})")
            
            if opt.phys_schedule_mode == 'relative':
                start_abs = opt.epoch_count + opt.phys_warmup_start
                end_abs = opt.epoch_count + opt.phys_warmup_end
                decay_start_abs = opt.epoch_count + opt.phys_decay_start
                decay_end_abs = opt.epoch_count + opt.phys_decay_end
                
                print(f" Schedule Mode   : Relative")
                print(f" Current Start   : Epoch {opt.epoch_count}")
                print(f" Relative Plan   : {opt.phys_warmup_start}e (start) -> {opt.phys_warmup_end}e (max) -> {opt.phys_decay_start}e (decay) -> {opt.phys_decay_end}e (stop)")
                print(f" Effective Epochs: {start_abs}e (start) -> {end_abs}e (max) -> {decay_start_abs}e (decay) -> {decay_end_abs}e (stop)")
            else:
                print(f" Schedule Mode   : Absolute")
                print(f" Absolute Plan   : {opt.phys_warmup_start}e (start) -> {opt.phys_warmup_end}e (max) -> {opt.phys_decay_start}e (decay) -> {opt.phys_decay_end}e (stop)")
                
            print(f" Velocity Kernels: {self.kernel_list}")
            print("---------------------------------------------")

            # === Preload Ground Truth Masks ===
            self.mask_cache = {}
            if os.path.exists(opt.mask_dataroot):
                from tqdm import tqdm
                mask_files = [f for f in os.listdir(opt.mask_dataroot) if f.lower().endswith(('.png', '.jpg', '.tif'))]
                print(f"Preloading masks to GPU memory...")
                for fname in tqdm(mask_files, desc="Loading Masks"):
                    fpath = os.path.join(opt.mask_dataroot, fname)
                    mask_cv = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                    if mask_cv is not None:
                        # Convert to 0/1 float tensor
                        mask_cv = (mask_cv > 127).astype(np.float32)
                        mask_tensor = torch.from_numpy(mask_cv).unsqueeze(0).to(self.device)
                        
                        file_id = os.path.splitext(fname)[0]
                        self.mask_cache[file_id] = mask_tensor
                        self.mask_cache[file_id.replace('_mask', '')] = mask_tensor
                print(f"Masks successfully preloaded: {len(mask_files)} images.")
            else:
                print(f"Warning: Mask dataroot does not exist: {opt.mask_dataroot}")

        # Loss names (for logging and display)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'physics', 'physics_weight']
        
        # Visual names for Visdom and HTML display
        # Cycle A: real_A -> fake_B -> rec_A | Masks: GT, Fake, Blur_Sim
        # Cycle A: real_A -> fake_B -> rec_A | Masks: GT, Fake, Blur_Sim
        visual_names_A = ['real_A', 'vis_gt_mask', 'vis_blur_sim', 'fake_B','vis_fake_mask', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B

        # Define networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.isTrain:
            # Match corresponding GT mask
            mask_list = []
            target_h, target_w = self.real_A.shape[2], self.real_A.shape[3]
            
            for img_path in self.image_paths:
                basename = os.path.basename(img_path)
                file_id = os.path.splitext(basename)[0]
                
                # Match mask using multiple filename key formats
                keys_to_try = [file_id, file_id + '_mask', basename]
                mask_tensor = None
                for k in keys_to_try:
                    if k in self.mask_cache:
                        mask_tensor = self.mask_cache[k]
                        break
                
                if mask_tensor is None:
                    # Fallback to zero mask if not found (loss will be zero)
                    mask_tensor = torch.zeros((1, target_h, target_w), device=self.device)
                else:
                    # Spatial resolution calibration
                    if mask_tensor.shape[1] != target_h or mask_tensor.shape[2] != target_w:
                         mask_tensor = torch.nn.functional.interpolate(
                            mask_tensor.unsqueeze(0), size=(target_h, target_w), mode='nearest'
                        ).squeeze(0)
                
                mask_list.append(mask_tensor)
            
            self.gt_mask = torch.stack(mask_list)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def set_current_epoch(self, epoch):
        self.manual_epoch = epoch

    def get_current_epoch(self):
        # Prioritize manually specified epoch
        if hasattr(self, 'manual_epoch') and self.manual_epoch is not None:
             return self.manual_epoch

        # Retrieve current epoch: max between scheduler state and initial epoch_count
        internal_epoch = 0
        if len(self.schedulers) > 0:
            internal_epoch = self.schedulers[0].last_epoch + 1
        return max(internal_epoch, self.opt.epoch_count)

    def compute_dynamic_weight(self, epoch_override=None):
        # 0. Prioritize epoch passed from optimize_parameters
        if epoch_override is not None:
            epoch = epoch_override
        else:
            epoch = self.get_current_epoch()

        opt = self.opt
        
        # Scheduling mode calculation
        if opt.phys_schedule_mode == 'relative':
            effective_epoch = max(0, epoch - opt.epoch_count)
        else:
            effective_epoch = epoch

        weight = 0.0
        phase = "Unknown"

        # 1. Pre-warmup phase
        if effective_epoch < opt.phys_warmup_start:
            weight = 0.0
            phase = "Pre-Warmup"
        
        # 2. Warmup phase
        elif effective_epoch < opt.phys_warmup_end:
            denom = opt.phys_warmup_end - opt.phys_warmup_start
            if denom <= 0:
                weight = 1.0
            else:
                weight = (effective_epoch - opt.phys_warmup_start) / denom
            phase = "Warmup"
        
        # 3. Sustain phase
        elif effective_epoch < opt.phys_decay_start:
            weight = 1.0
            phase = "Sustain"
        
        # 4. Linear decay phase
        elif effective_epoch < opt.phys_decay_end:
            denom = opt.phys_decay_end - opt.phys_decay_start
            if denom <= 0:
                weight = 0.0
            else:
                progress = (effective_epoch - opt.phys_decay_start) / denom
                weight = 1.0 - progress
            phase = "Decay"
        
        # 5. Decay ended
        else:
            weight = 0.0
            phase = "Ended"
            
        # Logging schedule status once per epoch
        if epoch_override is not None:
            if not hasattr(self, '_last_debug_epoch') or self._last_debug_epoch != epoch:
                self._last_debug_epoch = epoch
                
                print(f"\n{'='*45}")
                print(f"   Physics Weight Schedule (Epoch {epoch})")
                print(f"{'='*45}")
                print(f"1. Mode      : {opt.phys_schedule_mode.upper()}")
                if opt.phys_schedule_mode == 'relative':
                    print(f"   - Effective: {effective_epoch} ( = {epoch} - {opt.epoch_count} )")
                else:
                    print(f"   - Effective: {effective_epoch} (Global Epoch)")
    
                print(f"2. Stage Check:")
                print(f"   - [Sustain] < {opt.phys_decay_start} : {'True' if effective_epoch < opt.phys_decay_start else 'False'}")
                print(f"   - [Decay]   < {opt.phys_decay_end} : {'True' if effective_epoch < opt.phys_decay_end else 'False'}")
                
                print(f"3. Calculation:")
                if phase == "Decay":
                    denom = opt.phys_decay_end - opt.phys_decay_start
                    num = effective_epoch - opt.phys_decay_start
                    print(f"   - Formula  : 1.0 - (Current - Start) / (End - Start)")
                    print(f"   - Values   : 1.0 - ({effective_epoch} - {opt.phys_decay_start}) / ({opt.phys_decay_end} - {opt.phys_decay_start})")
                    print(f"   - Progress : {num/denom if denom!=0 else 0.0:.4f}")
                
                print(f"4. Result:")
                print(f"   - Current Stage : {phase}")
                print(f"   - Dynamic Weight: {weight:.6f}")
                print(f"{'='*45}\n")

        return weight

    def get_soft_mask(self, img):
        """
        Differentiable local adaptive soft thresholding with morphological refinement.
        Runs fully on GPU to maintain backpropagation gradient flow.
        """
        # 1. Normalize image from [-1, 1] to [0, 1]
        img_norm = (img + 1) / 2.0

        # 2. Gaussian pre-smoothing to suppress high-frequency GAN noise
        denoise_k = self.opt.mask_denoise_k
        if denoise_k % 2 == 0:
            denoise_k += 1
        if denoise_k >= 3:
            img_smooth = torch.nn.functional.avg_pool2d(
                img_norm, kernel_size=denoise_k, stride=1,
                padding=denoise_k // 2, count_include_pad=False
            )
        else:
            img_smooth = img_norm

        # 3. Global mean reference
        global_mean = img_smooth.mean(dim=(2, 3), keepdim=True)

        # 4. Difference = global_mean - smooth_pixel - C
        diff = global_mean - img_smooth - self.opt.adaptive_thresh_c

        # 5. Sigmoid soft binarization
        soft_mask = torch.sigmoid(self.opt.mask_sigmoid_gain * diff)

        # 6. Differentiable morphological refinement
        # --- (1) Opening (erosion followed by dilation): removes isolated noise and artifacts ---
        open_k = 5
        eroded = -torch.nn.functional.max_pool2d(-soft_mask, kernel_size=open_k, stride=1, padding=open_k // 2)
        opened_mask = torch.nn.functional.max_pool2d(eroded, kernel_size=open_k, stride=1, padding=open_k // 2)

        # --- (2) Closing (dilation followed by erosion): fills internal holes and connects gaps ---
        close_k = 11
        dilated = torch.nn.functional.max_pool2d(opened_mask, kernel_size=close_k, stride=1, padding=close_k // 2)
        final_mask = -torch.nn.functional.max_pool2d(-dilated, kernel_size=close_k, stride=1, padding=close_k // 2)
        
        return final_mask

    def calculate_projection_loss(self, pred, target):
        """
        Direct Y-axis projection loss: Max-pooling along the horizontal flow direction (X-axis)
        produces a 1D vertical profile representing cell occupancy per row, aligned via L1 loss.
        Enforces vertical boundary alignment and prevents vertical spatial distortion.
        """
        pred_proj_y = torch.max(pred, dim=3)[0]
        target_proj_y = torch.max(target, dim=3)[0]
        return torch.nn.functional.l1_loss(pred_proj_y, target_proj_y)

    def calculate_physics_loss(self):
        """
        Physics-informed consistency loss:
        Forward: Fake (Soft Mask) -> Motion blur elongation simulation -> Matched with GT Mask via Dice Loss.
        """
        # Fast exit if physics loss is disabled
        if self.opt.lambda_physics <= 1e-4:
            return torch.tensor(0.0, device=self.device)

        weight_ratio = self.compute_dynamic_weight()
        if weight_ratio <= 1e-3:
            if not hasattr(self, 'mask_blur_best'):
                self.mask_blur_best = torch.zeros_like(self.fake_B)
            return torch.tensor(0.0, device=self.device)

        # 1. Extract soft mask for fake_B
        fake_mask = self.get_soft_mask(self.fake_B)

        # 2. Multi-kernel motion blur scanning
        losses = []
        blurred_masks = []

        # GT mask requires no gradients
        target = self.gt_mask.detach()

        # Guard against empty GT masks
        if target.sum() < 5:
            self.mask_blur_best = torch.zeros_like(fake_mask)
            return torch.tensor(0.0, device=self.device)

        # Direct Y-axis projection loss between fake_mask and target mask
        loss_proj_direct = torch.tensor(0.0, device=self.device)
        if self.opt.weight_proj > 0:
            loss_proj_direct = self.calculate_projection_loss(fake_mask, target)

        for k_size in self.kernel_list:
            # Flow direction elongation along the X-axis
            # Step 1: Asymmetric zero-padding along the horizontal direction
            padded = torch.nn.functional.pad(
                fake_mask, (0, k_size - 1, 0, 0), mode='constant', value=0
            )

            # Step 2: 1D horizontal max pooling with kernel size (1, k_size)
            blur_sim = torch.nn.functional.max_pool2d(
                padded,
                kernel_size=(1, k_size),
                stride=1,
                padding=0
            )

            blurred_masks.append(blur_sim)

            # Dice Loss: 1 - (2 * intersection / (union + smooth))
            intersection = (blur_sim * target).sum(dim=(2, 3))
            union = blur_sim.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            dice = 1.0 - (2. * intersection + 1e-5) / (union + 1e-5)

            total_k = dice.mean() * self.opt.weight_dice
            losses.append(total_k)

        # 4. Winner-take-all: Select the velocity kernel with minimal Dice loss
        losses_stack = torch.stack(losses)
        best_idx = torch.argmin(losses_stack)
        loss_min = losses_stack[best_idx]

        # Cache the best simulated mask for visualization
        self.mask_blur_best = blurred_masks[best_idx].detach()

        # Combined loss = Best Dice Loss + direct Y-axis projection loss
        total_loss = loss_min + loss_proj_direct * self.opt.weight_proj
        return total_loss * self.opt.lambda_physics * weight_ratio

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity Loss
        if lambda_idt > 0:
            self.loss_idt_A = self.criterionIdt(self.netG_A(self.real_B), self.real_B) * lambda_B * lambda_idt
            self.loss_idt_B = self.criterionIdt(self.netG_B(self.real_A), self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN Loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Cycle Loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Physics Loss (Only for G_A: real_A -> fake_B)
        self.loss_physics = self.calculate_physics_loss()

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_physics
        self.loss_G.backward()

    def optimize_parameters(self, epoch=None):
        """ Calculate losses, gradients, and update network weights """
        self.forward()

        # Dynamic weight synchronization for current epoch
        w = self.compute_dynamic_weight(epoch_override=epoch)
        self.loss_physics_weight = torch.tensor(w, device=self.device)

        # 1. Update Generators (G_A and G_B)
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # 2. Update Discriminators (D_A and D_B)
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def backward_D_basic(self, netD, real, fake):
        """ Calculate GAN loss for discriminator """
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    
    def compute_visuals(self):
        BaseModel.compute_visuals(self)
        # 1. Ground truth mask visualization
        if hasattr(self, 'gt_mask'):
            self.vis_gt_mask = self.gt_mask * 2.0 - 1.0
        else:
            self.vis_gt_mask = torch.zeros_like(self.real_A)
        
        # 2. Fake mask visualization
        with torch.no_grad():
            soft = self.get_soft_mask(self.fake_B)
            self.vis_fake_mask = (soft > 0.5).float() * 2.0 - 1.0
        
        # 3. Best simulated motion blur mask
        if hasattr(self, 'mask_blur_best'):
            self.vis_blur_sim = (self.mask_blur_best > 0.5).float() * 2.0 - 1.0
        else:
            self.vis_blur_sim = torch.zeros_like(self.real_A)
            
        # Overlay vertical bounding lines (Red) from GT mask onto visualization images
        try:
            # 1. Identify connected components in GT mask to compute vertical bounds
            mask_np = (self.gt_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
            num_labels, labels_im = cv2.connectedComponents(mask_np)
            
            y_lines = []
            for i in range(1, num_labels):
                y_indices, _ = np.where(labels_im == i)
                if len(y_indices) > 0:
                    y_lines.append(np.min(y_indices)) # Top bound
                    y_lines.append(np.max(y_indices)) # Bottom bound
            
            # 2. Draw lines on tensor image
            def draw_lines(img_t, y_list):
                if img_t.shape[1] == 1:
                    res = img_t.repeat(1, 3, 1, 1)
                else:
                    res = img_t.clone()
                
                # Draw lines in Red [1, -1, -1] in [-1, 1] range
                for y in y_list:
                    if 0 <= y < res.shape[2]:
                        res[0, 0, y, :] = 1.0
                        res[0, 1, y, :] = -1.0
                        res[0, 2, y, :] = -1.0
                return res

            # 3. Apply to target visualization panels
            target_names = ['real_A', 'vis_gt_mask', 'vis_blur_sim', 'fake_B', 'vis_fake_mask', 'rec_A']
            for name in target_names:
                if hasattr(self, name):
                    origin_img = getattr(self, name)
                    drawn_img = draw_lines(origin_img, y_lines)
                    setattr(self, name, drawn_img)
                    
        except Exception as e:
            print(f"Draw lines warning: {e}")
            pass