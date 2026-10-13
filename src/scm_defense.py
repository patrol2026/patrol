import sys
import os
import torch
import transformers

def load_scm_monitor(checkpoint_path, scm_model_dir, device="cuda", dtype=torch.bfloat16):
    sys.path.insert(0, scm_model_dir)
    from modeling_qwen4dual_2CE_w_logic import QwenForDualTask
    tok = transformers.AutoTokenizer.from_pretrained(checkpoint_path)
    model = QwenForDualTask.from_pretrained(checkpoint_path, torch_dtype=dtype).to(device).eval()
    return model, tok

class SCMStopper(transformers.StoppingCriteria):
    def __init__(self, scm_model, scm_tok, target_tok, user_prompt, prompt_len,
                 delay_k, theta, check_interval, device):
        self.scm = scm_model
        self.stok = scm_tok
        self.ttok = target_tok
        self.prompt_len = prompt_len
        self.k = delay_k
        self.theta = theta
        self.ci = max(1, check_interval)
        self.device = device
        self.q_prompt = scm_tok(user_prompt, add_special_tokens=False)["input_ids"]
        self.n_calls = 0
        self.aborted = False
        self.stop_pos = -1
        self.n_flagged = 0

    @torch.no_grad()
    def __call__(self, input_ids, scores, **kw):
        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len <= 0 or gen_len % self.ci != 0:
            return False
        resp_text = self.ttok.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        if not resp_text.strip():
            return False
        q_resp = self.stok(resp_text, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([self.q_prompt + q_resp], device=self.device)
        p_harm = torch.softmax(self.scm(input_ids=ids)["token_logits"][0], -1)[:, 1]
        n_flag = int((p_harm[len(self.q_prompt):] >= self.theta).sum())
        self.n_calls += 1
        if n_flag >= self.k:
            self.aborted = True
            self.stop_pos = gen_len
            self.n_flagged = n_flag
            return True
        return False

@torch.no_grad()
def scm_generate(input_ids, target_model, gen_config, tokenizer,
                 scm_model, scm_tok, user_prompt, delay_k, theta, check_interval, device):
    prompt_len = input_ids.shape[1]
    stopper = SCMStopper(scm_model, scm_tok, tokenizer, user_prompt, prompt_len,
                         delay_k, theta, check_interval, device)
    out = target_model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        generation_config=gen_config,
        stopping_criteria=transformers.StoppingCriteriaList([stopper]),
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, prompt_len:]
    response = tokenizer.decode(gen, skip_special_tokens=True)
    stats = {
        "aborted": bool(stopper.aborted),
        "early_stop_position": int(stopper.stop_pos),
        "n_monitor_calls": int(stopper.n_calls),
        "n_flagged_tokens": int(stopper.n_flagged),
        "n_output_tokens": int(gen.shape[0]),
        "delay_k": delay_k,
        "theta": theta,
        "check_interval": check_interval,
    }
    return response, stats
