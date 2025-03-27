# inference/infer.py

import torch


def generate_text(model, tokenizer, prompt, device, max_length=100):
    """
    Autoregressively generate text given an input prompt.

    Args:
        model (nn.Module): The RMKVModel.
        tokenizer: Tokenizer instance with encode and decode methods.
        prompt (str): Input text prompt.
        device (torch.device): Computation device.
        max_length (int): Maximum number of tokens to generate.

    Returns:
        str: The generated text.
    """
    model.eval()
    with torch.no_grad():
        # Encode the prompt using the tokenizer.
        input_ids = tokenizer.encode(prompt)
        # Convert to tensor and add batch dimension.
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        generated = input_ids
        for _ in range(max_length):
            logits = model(generated)
            # Select the logits for the last generated token.
            next_token_logits = logits[:, -1, :]
            # Greedy decoding: choose the token with the highest logit.
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            # Assume token ID 0 is the end-of-sequence marker.
            if next_token.item() == 0:
                break

    # Decode the generated token IDs back into a string.
    output_text = tokenizer.decode(generated[0].tolist())
    return output_text
