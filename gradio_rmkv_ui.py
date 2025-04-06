import gradio as gr
import torch
from model.rmkv import RMKVModel
from data.tokenizer import RemarkableTokenizer
from config import PATHS

# python gradio_rmkv_ui.py

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RemarkableTokenizer(load_path=f"{PATHS['tokenizer_dir']}/tokenizer.json")
model = RMKVModel(tokenizer.vocab_size_actual)
model.load_state_dict(torch.load("checkpoints/rmkv_finetune_final.pt", map_location=device))
model.to(device)
model.eval()

# Inference function
def generate(prompt, max_length=256, temperature=0.8, top_p=0.95):
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_tensor)
            next_token_logits = output[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

            # Apply nucleus (top-p) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative_probs > top_p
            if torch.any(cutoff):
                cutoff_idx = torch.argmax(cutoff).item() + 1
                sorted_probs = sorted_probs[:, :cutoff_idx]
                sorted_indices = sorted_indices[:, :cutoff_idx]
                probs = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token == tokenizer.pad_token_id:
                break
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1
            )

    return tokenizer.decode(input_tensor[0].tolist())

# Gradio interface
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", lines=4, placeholder="Ask something..."),
        gr.Slider(32, 1024, value=256, step=16, label="Max Length"),
        gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p (nucleus sampling)")
    ],
    outputs=gr.Textbox(label="Response", lines=8),
    title="RMKV Model Playground",
    description="Try out your fine-tuned RMKV model."
)

demo.launch()
