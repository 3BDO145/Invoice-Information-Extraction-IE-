**Invoice Information Extraction (IE) Notebook**

This notebook fine-tunes a **LayoutLMv3** model for token classification (NER) using **LoRA** adapters and demonstrates inference with visualization of predicted entities on document images.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Inference & Visualization](#inference--visualization)
7. [Example Outputs](#example-outputs)
8. [Notes](#notes)

---

## Overview

This project uses **LoRA** (Low‑Rank Adaptation) to efficiently fine-tune a pre-trained LayoutLMv3 model for a document NER task. The notebook covers:

* Configuring LoRA adapters for efficient training
* Fine-tuning on a labeled document dataset
* Evaluating train, validation, and test metrics
* Running inference on single/multiple samples
* Visualizing predictions by drawing colored bounding boxes and labels on images

---

## Requirements

* Python 3.8+
* PyTorch 1.12+
* Transformers 4.30+
* PEFT 0.4.0+ (`pip install peft`)
* Datasets (Hugging Face)
* Pillow, matplotlib, pandas

```bash
pip install torch transformers peft datasets pillow matplotlib pandas
```

---

## Setup

1. Clone or download this repository.
2. Place your LoRA checkpoint folder (e.g. `nlu-lora-checkpoint-2000`) under `/kaggle/input/` or update the path in the inference cell.
3. Update the `id2label` and `label2id` mappings to match your dataset labels.

---

## Training

LoRA configuration:

```python
from peft import LoraConfig, get_peft_model
# LoRA adapter parameters
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query","value"],
    lora_dropout=0.1,
    bias="none",
    task_type="TOKEN_CLS"
)

# Load base model and attach LoRA adapters
en_model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model = get_peft_model(en_model, lora_config)
```

Training arguments with early stopping:

```python
from transformers import Trainer, TrainingArguments
from transformers.integrations import EarlyStoppingCallback

dirs = "LayoutLMv3-LoRA"
training_args = TrainingArguments(
    output_dir=dirs,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=100,
    eval_steps=100,
    logging_steps=50,
    learning_rate=2e-4,
    warmup_steps=500,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_strategy="steps",
    eval_strategy="steps",
    report_to="none",
    push_to_hub=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=processor,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
# Start training
trainer.train()
```

### Expected Training Log

```
***** Running training *****
  Num examples = 3200
  Num Epochs = 10
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed) = 8
  Total optimization steps = 4000
...
Step 50/4000 - loss: 0.2432 - f1: 0.8214 - lr: 2.0e-04
Step 100 - eval_loss: 0.1987 - eval_f1: 0.8450  
Saving model checkpoint to LayoutLMv3-LoRA/checkpoint-100
...
Training completed. Best model saved at checkpoint-800
```

---

## Evaluation

**Sample metrics output (recorded every 100 training steps):**

!\[Training Metrics]\(/mnt/data/Screenshot 2025-05-12 034509.png)

| Step | Training Loss | Validation Loss | Precision | Recall   | F1       | Accuracy |
| ---- | ------------- | --------------- | --------- | -------- | -------- | -------- |
| 100  | 1.219300      | 0.776498        | 0.789689  | 0.781377 | 0.785511 | 0.803751 |
| 200  | 0.449500      | 0.303259        | 0.887449  | 0.887449 | 0.887449 | 0.912626 |
| 300  | 0.372000      | 0.272342        | 0.884459  | 0.880162 | 0.882305 | 0.914456 |
| 400  | 0.324100      | 0.363914        | 0.884992  | 0.878543 | 0.881755 | 0.897530 |
| 500  | 0.317800      | 0.275339        | 0.891827  | 0.901215 | 0.896496 | 0.928179 |
| 600  | 0.328200      | 0.266067        | 0.917006  | 0.912551 | 0.914773 | 0.934126 |
| 700  | 0.193700      | 0.192058        | 0.936791  | 0.936032 | 0.936412 | 0.954712 |
| 800  | 0.151100      | 0.219024        | 0.935795  | 0.944130 | 0.939944 | 0.955169 |
| 900  | 0.085300      | 0.196777        | 0.940467  | 0.946559 | 0.943503 | 0.959286 |
| 1000 | 0.053800      | 0.151231        | 0.950000  | 0.953846 | 0.951919 | 0.966606 |

---

## Inference & Visualization

To load the LoRA‐tuned model and visualize predictions on a document image:

```python
import torch
from transformers import AutoProcessor
from peft import PeftModel
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base and overlay LoRA weights
base = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device)
model = PeftModel.from_pretrained(base, dirs).to(device)
model.eval()
```

Run inference on one sample:

```python
sample = dataset['test'][0]
encoding = processor(
    sample['image'], sample['words'], boxes=sample['bboxes'],
    padding='max_length', truncation=True, return_tensors='pt'
).to(device)
outputs = model(**encoding)
preds = outputs.logits.argmax(dim=2).squeeze().cpu().tolist()
```

Draw boxes & labels:

```python
from PIL import ImageDraw, ImageFont
image = sample['image'].convert("RGB")
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
for tok_idx, word_idx in enumerate(encoding.word_ids()):
    if word_idx is None: continue
    box = sample['bboxes'][word_idx]
    lbl = id2label[preds[tok_idx]].split('-',1)[-1]
    box_px = unnormalize_box(box, *image.size)
    draw.rectangle(box_px, outline="red", width=2)
    draw.text((box_px[0]+2, box_px[1]-24), lbl, fill="red", font=font)
```

---

## Example Outputs
### Visualized Receipt

Below is an excerpt from the test set with predicted bounding boxes drawn in red and labels above each box:

![Annotated Receipt](https://raw.githubusercontent.com/3BDO145/Invoice-Information-Extraction-IE-/main/Screenshot%202025-05-12%20034630.png)

 *Red boxes indicate detected entities with their predicted labels.*

### DataFrame Snippet

```python
    word    true_label              pred_label
0   J.STB   I-menu.nm              I-menu.nm
1   PROMO   I-menu.nm              I-menu.nm
2   17500   I-menu.price           I-menu.price
3   Y.B.BAT I-menu.nm              I-menu.nm
4   46000   I-menu.price           I-menu.price
5   Y.BASO  I-menu.nm              I-menu.nm
6   PROM    I-menu.nm              I-menu.nm
7   27500   I-menu.price           I-menu.price
8   TOTAL   I-total.total_price    I-total.total_price
9   91000   I-total.total_price    I-total.total_price
10  CASH    I-total.cashprice      I-total.cashprice
11  91000   I-total.cashprice      I-total.cashprice
```


## Notes

* Adjust `r`, `lora_alpha`, and `target_modules` in `LoraConfig` for different trade‑offs.
* Ensure your dataset boxes are in LayoutLMv3’s 0–1000 coordinate scale.
* Always call `model.eval()` before inference to disable dropout layers.

---

*Author: Abdelrhman Kaled*

*Last updated: May 2025*
