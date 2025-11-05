from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

def generate_answer(query, best_answer, model_name="GajahTerbang/llama3-8B-4BitIndoAlpacaFineTuned"):
    alpaca_prompt = (
        "Instruksi:\n{0}\n\n"
        "Masukan:\n{1}\n\n"
        "Jawaban:\n"
    )

    prompt = alpaca_prompt.format(
        f"Tolong jawab pertanyaan ini dengan sopan dan jelas:\n{query}",
        best_answer
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    streamer = TextStreamer(tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("\n=== Jawaban Natural dari LLaMA ===")
    _ = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=250,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1
    )
