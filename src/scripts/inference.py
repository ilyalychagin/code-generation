import torch
import json
from pathlib import Path
from tqdm.auto import tqdm

@torch.no_grad()
def inference(model, test_loader, device, tokenizer, max_new_tokens=128):

    model.eval()

    answer = []
    progress = tqdm(
        test_loader,
        desc="Inference",
        unit="batch",
        dynamic_ncols=True,
        mininterval=2.0,
    )
    for batch in progress:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }
        output_tokens = model.generate(**batch, max_new_tokens=max_new_tokens)
        outputs = tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        answer.extend({"code": code} for code in outputs)
        progress.set_postfix(samples=len(answer), refresh=False)
    
    return answer

def write_answer(model, test_loader, device, tokenizer, output_path="pass1_questions.jsonl", max_new_tokens=128):
    answer = inference(model, test_loader, device, tokenizer, max_new_tokens=max_new_tokens)
    
    assert len(answer) == len(test_loader.dataset)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for item in answer:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    tqdm.write(f"Saved {len(answer)} predictions to {output_path}")
