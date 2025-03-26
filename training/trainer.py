# training/trainer.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import TRAINING_CONFIG, PATHS
from model.rmkv import RMKVModel
from training.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, train_dataset, device):
        self.device = device
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.dataloader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True)
        self.grad_accum_steps = TRAINING_CONFIG["gradient_accumulation_steps"]
        self.max_grad_norm = TRAINING_CONFIG["max_grad_norm"]
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.start_epoch = 0

    def _build_optimizer(self):
        # Currently using AdamW optimizer
        return optim.AdamW(
            self.model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )

    def _build_scheduler(self):
        # Using a simple cosine scheduler
        total_steps = len(self.dataloader) * TRAINING_CONFIG["num_epochs"] // self.grad_accum_steps
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps)
        return scheduler

    def train(self):
        self.model.train()
        num_epochs = TRAINING_CONFIG["num_epochs"]
        accumulation_steps = self.grad_accum_steps
        total_loss = 0.0

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start_time = time.time()
            for step, batch in enumerate(self.dataloader):
                # Batch is a dictionary with 'input_ids' and optionally 'labels'.
                # Here we use input_ids as both input and target (causal LM training).
                input_ids = batch["input_ids"].to(self.device)
                # For causal LM, target is input_ids shifted left.
                labels = input_ids.clone().to(self.device)

                outputs = self.model(input_ids)
                # outputs: (batch, seq_len, vocab_size)
                # Shift logits and labels for loss computation.
                logits = outputs[:, :-1, :].contiguous()
                target = labels[:, 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.size(-1)), target.view(-1))
                loss = loss / accumulation_steps  # Normalize loss for accumulation

                loss.backward()
                total_loss += loss.item()

                if (step + 1) % accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if (step + 1) % (accumulation_steps * 10) == 0:
                        print(f"Epoch [{epoch + 1}/{num_epochs}] Step [{step + 1}/{len(self.dataloader)}] "
                              f"Loss: {total_loss / (step + 1):.4f}")

            epoch_duration = time.time() - epoch_start_time
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_duration:.2f}s, average loss: {avg_loss:.4f}")

            # Save checkpoint at the end of each epoch
            ckpt_path = os.path.join(PATHS["checkpoint_dir"], f"rmkv_epoch_{epoch + 1}.pt")
            save_checkpoint(self.model, self.optimizer, epoch, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    # For testing purposes, create a dummy dataset and run the trainer.
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000, seq_len=128, vocab_size=10000):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Random sequence of token ids.
            sample = torch.randint(0, self.vocab_size, (self.seq_len,))
            return {"input_ids": sample}


    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy dataset and model
    dummy_dataset = DummyDataset()
    vocab_size = 10000
    model = RMKVModel(vocab_size)

    trainer = Trainer(model, dummy_dataset, device)
    trainer.train()
