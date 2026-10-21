from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained('./nd_dpo_merged_local', torch_dtype=torch.float16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('./nd_dpo_merged_local')

SYSTEM_PROMPT = (
    "Act as a low-demand, neurodiversity-affirming clinical companion grounded in "
    "Linda K. Murphy's BERMED Declarative Language, Richard Erskine's Schizoid-Process "
    "Attunement to Silence, and Johan Pahnke's NeuroACT. Never ask questions, never give "
    "imperative commands, and never pressure the user to socialize."
)

test_prompts = [
    "Been staring at the ceiling for an hour, cant make myself get up.",
    "My phone has 40 unread messages and I havent looked at it in three days.",
    "Someone invited me to a party this weekend and I dont know how to say no.",
]

for p in test_prompts:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]
    inputs = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    reply = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"PROMPT: {p}\nREPLY: {reply}\n{'-'*60}")
