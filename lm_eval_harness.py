from lm_eval.base import BaseLM
import torch
from model.rmkv import RMKVModel
from data.tokenizer import RemarkableTokenizer
from config import PATHS

# documentation here: https://github.com/EleutherAI/lm-evaluation-harness

# pip install lm-eval

# pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git


# lm_eval \
#   --model custom \
#   --model_args 'wrapper=path.to.RMKVWrapper,checkpoint_path=checkpoints/rmkv_finetune_final.pt' \
#   --tasks truthfulqa_mc \
#   --device cuda \
#   --batch_size 1 \
#   --output_path results/truthfulqa.json

# lm_eval \
#   --model huggingface \
#   --model_args pretrained=none,tokenizer=none,adapter=eval_harness_adapter.RMKVWrapper,checkpoint_path=checkpoints/rmkv_finetune_final.pt \
#   --tasks truthfulqa_mc \
#   --device cuda \
#   --batch_size 1 \
#   --output_path results/truthfulqa.json

class RMKVWrapper(BaseLM):
    def __init__(self, checkpoint_path, max_length, device=None):
        super().__init__()
        self.tokenizer = RemarkableTokenizer(load_path=f"{PATHS['tokenizer_dir']}/tokenizer.json")
        self.model = RMKVModel(self.tokenizer.vocab_size_actual)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device or "cpu"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    @property
    def eot_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def max_length(self):
        return self._max_length

    @max_length.setter
    def max_length(self, val):
        self._max_length = val

    @property
    def max_gen_toks(self):
        return 128

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size_actual

    @property
    def batch_size(self):
        return 1

    def tok_encode(self, string):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """Accepts a tensor of shape [batch, seq] and returns logits [batch, seq, vocab]"""
        with torch.no_grad():
            return self.model(inps.to(self.device)).cpu()

    def _model_generate(self, context, max_length, eos_token_id):
        """Greedy decoding for eval tasks like multiple choice"""
        generated = context
        for _ in range(max_length):
            logits = self._model_call(generated)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if next_token.item() == eos_token_id:
                break
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    def loglikelihood(self, requests):
        results = []
        for context, continuation in requests:
            input_ids = self.tok_encode(context + continuation)
            context_ids = self.tok_encode(context)

            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            logits = self._model_call(input_tensor)

            # Compute token-level log probabilities of the continuation
            cont_offset = len(context_ids)
            targets = torch.tensor(input_ids[cont_offset:], dtype=torch.long)
            logits = logits[0, cont_offset - 1 : -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(targets)), targets]
            results.append((float(selected.sum()), True))
        return results

    def greedy_until(self, requests):
        generations = []
        for context, stop in requests:
            input_ids = self.tok_encode(context)
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            output = self._model_generate(input_tensor, max_length=64, eos_token_id=self.tokenizer.pad_token_id)
            generations.append(self.tok_decode(output[0].tolist()))
        return generations
