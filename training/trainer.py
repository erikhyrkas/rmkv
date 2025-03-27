# training/trainer.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import TRAINING_CONFIG, PATHS
from training.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, train_dataset, device):
        self.device = device
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        #             num_workers=0,
        #             persistent_workers=False
        self.grad_accum_steps = TRAINING_CONFIG["gradient_accumulation_steps"]
        self.max_grad_norm = TRAINING_CONFIG["max_grad_norm"]
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.start_epoch = 0
        self.global_step = 0

    def _build_optimizer(self):
        # Using AdamW optimizer
        return optim.AdamW(
            self.model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )

    def _build_scheduler(self):
        # Calculate total steps properly
        steps_per_epoch = len(self.dataloader) // self.grad_accum_steps
        total_steps = steps_per_epoch * TRAINING_CONFIG["num_epochs"]
        # Using a cosine scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        return scheduler

    def train(self):
        self.model.train()
        num_epochs = TRAINING_CONFIG["num_epochs"]

        if self.start_epoch > num_epochs:
            print("Already reached max epochs.")
            return

        # Initialize optimizer - don't recreate for each epoch
        self.optimizer.zero_grad()

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0

            print(f"Epoch [{epoch + 1}/{num_epochs}] starting. LR: {self.scheduler.get_last_lr()[0]:.6e}")

            for step, batch in enumerate(self.dataloader):
                input_ids = batch["input_ids"].to(self.device)
                labels = input_ids.clone().to(self.device)

                outputs = self.model(input_ids)
                logits = outputs[:, :-1, :].contiguous()
                target = labels[:, 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                # Don't divide loss by accum_steps - PyTorch handles this correctly with backward()
                loss = loss_fct(logits.view(-1, logits.size(-1)), target.view(-1))

                # Scale the loss for gradient accumulation - this is the correct way
                scaled_loss = loss / self.grad_accum_steps
                scaled_loss.backward()

                # Track full loss for logging
                epoch_loss += loss.detach().item()

                if step == 0 and epoch == self.start_epoch:
                    print(f"Epoch [{epoch + 1}] - Initial loss: {loss.item():.4f}")

                # Only update after accumulating enough gradients or at the end of the epoch
                if (step + 1) % self.grad_accum_steps == 0 or step == len(self.dataloader) - 1:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Step optimizer
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Step scheduler
                    self.scheduler.step()
                    self.global_step += 1

                    if (step + 1) % (self.grad_accum_steps * 10) == 0:
                        current_lr = self.scheduler.get_last_lr()[0]
                        avg_loss = epoch_loss / (step + 1)
                        print(f"Epoch [{epoch + 1}/{num_epochs}] Step [{step + 1}/{len(self.dataloader)}] "
                              f"Loss: {avg_loss:.4f} LR: {current_lr:.6e} GlobalStep: {self.global_step}")

            # Ensure proper CUDA synchronization for timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            epoch_duration = time.time() - epoch_start_time
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_duration:.2f}s, average loss: {avg_loss:.4f}")

            # Save checkpoint at the end of each epoch
            ckpt_path = os.path.join(PATHS["checkpoint_dir"], f"rmkv_epoch_{epoch + 1}.pt")
            save_checkpoint(self.model, self.optimizer, epoch, self.global_step, ckpt_path)
