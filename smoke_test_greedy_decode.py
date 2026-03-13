from __future__ import annotations

import argparse
import torch

from generate_news_llm_responses import (
    DEFAULT_MODEL_NAME_OR_PATH,
    generate_responses_batched,
    load_model_and_tokenizer,
    render_chat_prompt,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the custom greedy decode path.")
    parser.add_argument(
        "--model-name-or-path",
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help=f"Model name or local path. Default: {DEFAULT_MODEL_NAME_OR_PATH}",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["auto", "cpu", "cuda"],
        help="Torch device to use.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
        help="Maximum number of generated tokens per prompt.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="User prompt to test. Can be passed multiple times.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a concise assistant.",
        help="System prompt used to render the chat prompt.",
    )
    parser.add_argument(
        "--store-pre-response-embedding",
        action="store_true",
        help="Also print the extracted pre-response embedding shape.",
    )
    return parser.parse_args()


@torch.inference_mode()
def generate_with_official_generate(
    rendered_prompts: list[str],
    tokenizer,
    model,
    device,
    *,
    max_new_tokens: int,
) -> list[str]:
    if not rendered_prompts:
        return []

    encoded = tokenizer(
        rendered_prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    prompt_width = int(encoded["input_ids"].shape[1])
    generated_only = generated[:, prompt_width:]
    return tokenizer.batch_decode(generated_only, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer, model = load_model_and_tokenizer(args.model_name_or_path, device)

    user_prompts = args.prompt or [
        "Finish this phrase with a short answer: The capital of France is",
        "Answer with one short finance word: market",
    ]
    rendered_prompts = [
        render_chat_prompt(args.system_prompt, user_prompt, tokenizer) for user_prompt in user_prompts
    ]

    responses, pre_response_embeddings = generate_responses_batched(
        prompts=rendered_prompts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=args.max_new_tokens,
        store_pre_response_embedding=bool(args.store_pre_response_embedding),
        pre_response_embedding_dtype="float16",
    )
    official_responses = generate_with_official_generate(
        rendered_prompts=rendered_prompts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    for idx, (user_prompt, custom_response, official_response) in enumerate(
        zip(user_prompts, responses, official_responses, strict=True),
        start=1,
    ):
        print(f"[prompt {idx}] {user_prompt}")
        print(f"[custom response {idx}] {custom_response}")
        print(f"[official response {idx}] {official_response}")
        print(f"[match {idx}] {custom_response == official_response}")
        print()

    if pre_response_embeddings is not None:
        print(f"pre_response_embeddings.shape={pre_response_embeddings.shape}")
        print(f"pre_response_embeddings.dtype={pre_response_embeddings.dtype}")


if __name__ == "__main__":
    main()
