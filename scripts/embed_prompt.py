import argparse

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Embed a prompt")
    parser.add_argument("prompt", help="Text prompt to embed")
    parser.add_argument(
        "-o",
        "--output",
        default="prompt.safetensors",
        help="Output safetensors file (default: prompt.safetensors)",
    )
    args = parser.parse_args()

    print(f"Embedding prompt: `{args.prompt}`")

    device = "cpu"

    model_id = "Tongyi-MAI/Z-Image-Turbo"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer"
    )
    text_encoder = AutoModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        dtype=torch.float16,
    )

    with torch.no_grad():
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        text_input = tokenizer(
            formatted_prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_input.input_ids.to(device)
        prompt_mask = text_input.attention_mask.to(device).bool()

        prompt_embed = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_mask,
            output_hidden_states=True,
        ).hidden_states[-2]
        embedding = prompt_embed[prompt_mask]

    save_file({"prompt": embedding.cpu().contiguous()}, args.output)


if __name__ == "__main__":
    main()
