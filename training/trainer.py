# training/trainer.py

import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from config import PRETRAIN_CONFIG, FINETUNE_CONFIG, PATHS, MODEL_CONFIG
from training.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, train_dataset, device, mode="pretrain"):
        self.device = device
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.mode = mode

        # Select the appropriate configuration based on mode
        self.config = PRETRAIN_CONFIG if mode == "pretrain" else FINETUNE_CONFIG

        # Set up logging
        self._setup_logging()
        self.log(f"Using {mode} configuration with {self.config['num_epochs']} epochs")

        self.dataloader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
        )

        self.grad_accum_steps = self.config["gradient_accumulation_steps"]
        self.max_grad_norm = self.config["max_grad_norm"]
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.start_epoch = 0
        self.global_step = 0

        # Initialize timing metrics
        self.step_times = []
        self.epoch_start_time = 0
        self.training_start_time = 0

    def _setup_logging(self):
        """Set up logging configuration."""
        # Generate a timestamp for the log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"rmkv_{self.mode}_{timestamp}.log"
        log_path = os.path.join(PATHS["logs_dir"], log_filename)

        # Configure file handler
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()  # This will duplicate to console, but we'll handle console printing separately
            ]
        )

        self.logger = logging.getLogger("RMKV_Trainer")
        self.log_file_path = log_path
        self.log(f"Logging initialized. Log file: {log_path}")

    def log(self, message, level="info"):
        """Log a message to both console and file."""
        # Log to file based on level
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)

    def _build_optimizer(self):
        # Using AdamW optimizer
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

    def _build_scheduler(self):
        # Calculate total steps properly
        steps_per_epoch = len(self.dataloader) // self.grad_accum_steps
        total_steps = steps_per_epoch * self.config["num_epochs"]
        # Using a cosine scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        return scheduler

    def update_mode(self, mode, reset_epochs=True):
        """Update training mode and reset optimizer/scheduler."""
        self.log(f"Switching training mode from {self.mode} to {mode}")
        self.mode = mode
        self.config = PRETRAIN_CONFIG if mode == "pretrain" else FINETUNE_CONFIG

        # Reset optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Reset epoch counter if specified
        if reset_epochs:
            self.start_epoch = 0
            self.log(f"Reset epoch counter. Will train for {self.config['num_epochs']} epochs.")

        # Reset timing metrics
        self.step_times = []
        self.epoch_start_time = 0
        self.training_start_time = 0

        # Setup new log file for the new mode
        self._setup_logging()

    def _format_time(self, seconds):
        """Format seconds into a human-readable string."""
        return str(datetime.timedelta(seconds=int(seconds)))

    def _estimate_remaining_time(self, current_epoch, current_step, total_steps_per_epoch):
        """Estimate remaining time based on average step time."""
        if not self.step_times:
            return "unknown"

        # Calculate remaining steps in current epoch
        remaining_steps_in_epoch = total_steps_per_epoch - current_step

        # Calculate remaining full epochs
        remaining_full_epochs = self.config["num_epochs"] - current_epoch - 1

        # Total remaining steps
        total_remaining_steps = remaining_steps_in_epoch + (remaining_full_epochs * total_steps_per_epoch)

        # Estimate using median of last 50 step times for stability
        recent_steps = self.step_times[-50:] if len(self.step_times) > 50 else self.step_times
        avg_step_time = sum(recent_steps) / len(recent_steps)

        return self._format_time(avg_step_time * total_remaining_steps)

    def train(self):
        self.model.train()
        num_epochs = self.config["num_epochs"]

        if self.start_epoch >= num_epochs:
            self.log(f"Already reached max epochs for {self.mode} mode ({self.start_epoch}/{num_epochs}).")
            return

        # Initialize optimizer - don't recreate for each epoch
        self.optimizer.zero_grad()

        # Record overall training start time
        self.training_start_time = time.time()
        total_steps_per_epoch = len(self.dataloader) // self.grad_accum_steps

        separator = f"{'-' * 80}"
        self.log(f"\n{separator}")
        self.log(f"Starting {self.mode} training for {num_epochs - self.start_epoch} epochs")
        self.log(f"Total steps per epoch: {total_steps_per_epoch}")
        self.log(f"Total training steps: {total_steps_per_epoch * (num_epochs - self.start_epoch)}")
        self.log(f"{separator}\n")

        for epoch in range(self.start_epoch, num_epochs):
            self.epoch_start_time = time.time()
            epoch_loss = 0.0
            step_start_time = time.time()

            self.log(
                f"\n{self.mode.capitalize()} Epoch [{epoch + 1}/{num_epochs}] starting. LR: {self.scheduler.get_last_lr()[0]:.6e}")

            # Reset step times for this epoch
            self.step_times = [] if epoch == self.start_epoch else self.step_times[-100:]

            for step, batch in enumerate(self.dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_masks = batch["attention_mask"].to(self.device)
                labels = input_ids.clone().to(self.device)

                outputs = self.model(input_ids, attention_masks)
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

                    # Calculate step time for completed batch
                    step_end_time = time.time()
                    step_time = step_end_time - step_start_time
                    self.step_times.append(step_time)
                    step_start_time = time.time()  # Reset for next step

                    # Effective steps accounting for gradient accumulation
                    effective_step = (step + 1) // self.grad_accum_steps

                    if (step + 1) % (self.grad_accum_steps * 10) == 0 or step == 0:
                        # Get timing information
                        elapsed_epoch_time = time.time() - self.epoch_start_time
                        elapsed_total_time = time.time() - self.training_start_time
                        est_epoch_remaining = self._estimate_remaining_time(
                            epoch, effective_step, total_steps_per_epoch
                        )

                        current_lr = self.scheduler.get_last_lr()[0]
                        avg_loss = epoch_loss / (step + 1)

                        # Calculate steps per second
                        steps_per_sec = effective_step / max(1, elapsed_epoch_time)
                        tokens_per_sec = steps_per_sec * self.config["batch_size"] * MODEL_CONFIG["max_seq_len"]

                        # Log step information
                        step_info = [
                            f"\n{self.mode.capitalize()} Epoch [{epoch + 1}/{num_epochs}] "
                            f"Step [{effective_step}/{total_steps_per_epoch}] ({step + 1}/{len(self.dataloader)})"
                            f" Loss: {avg_loss:.4f}",

                            f"  LR: {current_lr:.6e}",

                            f"  Speed: {steps_per_sec:.2f} steps/s ({tokens_per_sec:.2f} tokens/s)",

                            f"  Step time: {step_time:.2f}s | Avg step time: "
                            f"{sum(self.step_times[-10:]) / min(10, len(self.step_times[-10:])):.2f}s",

                            f"  Epoch progress: {(effective_step / total_steps_per_epoch) * 100:.1f}% | "
                            f"Elapsed: {self._format_time(elapsed_epoch_time)} | "
                            f"Remaining: {est_epoch_remaining}",

                            f"  Total training time: {self._format_time(elapsed_total_time)}"
                        ]

                        for line in step_info:
                            self.log(line)

            # Ensure proper CUDA synchronization for timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Epoch summary
            epoch_duration = time.time() - self.epoch_start_time
            avg_loss = epoch_loss / len(self.dataloader)

            # Calculate overall progress
            overall_progress = (epoch + 1 - self.start_epoch) / (num_epochs - self.start_epoch)
            elapsed_total_time = time.time() - self.training_start_time
            est_total_remaining = self._format_time(
                elapsed_total_time * (1 / overall_progress - 1)) if overall_progress > 0 else "unknown"

            # Log epoch summary
            epoch_summary = [
                f"\n{separator}",
                f"{self.mode.capitalize()} Epoch [{epoch + 1}/{num_epochs}] completed:",
                f"  Loss: {avg_loss:.4f}",
                f"  Duration: {self._format_time(epoch_duration)} ({len(self.dataloader) / epoch_duration:.1f} batches/s)",
                f"  Overall progress: {overall_progress * 100:.1f}%",
                f"  Total elapsed: {self._format_time(elapsed_total_time)}",
                f"  Estimated remaining: {est_total_remaining}",
                f"{separator}"
            ]

            for line in epoch_summary:
                self.log(line)

            # Save checkpoint at the end of each epoch
            ckpt_path = os.path.join(PATHS["checkpoint_dir"], f"rmkv_{self.mode}_epoch_{epoch + 1}.pt")
            latest_path = os.path.join(PATHS["checkpoint_dir"], f"rmkv_{self.mode}_latest.pt")

            # Save both the specific epoch checkpoint and update the latest checkpoint
            save_checkpoint(self.model, self.optimizer, epoch, self.global_step, ckpt_path)
            save_checkpoint(self.model, self.optimizer, epoch, self.global_step, latest_path)

            # For backward compatibility
            general_latest_path = os.path.join(PATHS["checkpoint_dir"], "rmkv_latest.pt")
            save_checkpoint(self.model, self.optimizer, epoch, self.global_step, general_latest_path)

            self.log(f"Saved checkpoints for epoch {epoch + 1}")
            self.log(f"{separator}\n")
