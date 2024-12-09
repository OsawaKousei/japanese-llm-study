import logging
from logging import Formatter, StreamHandler, getLogger

from model import BPRModel
from prepare import collate_fn, tokenizer, train_dataset, valid_dataset
from transformers import Trainer, TrainingArguments

# ログの設定
if __name__ == "__main__":
    logger = getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
else:
    logger = getLogger("__main__").getChild(__name__)

# パラメータ
training_args = TrainingArguments(
    output_dir="outputs_bpr",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    max_grad_norm=2.0,
    num_train_epochs=20,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,
    remove_unused_columns=False,
)

# モデルの初期化
model = BPRModel("cl-tohoku/bert-base-japanese-v3")

# Trainerの初期化
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# モデルのパラメータを連続的なメモリ領域に配置
# 以下のエラーを回避するために追記
# ValueError: You are trying to save a non contiguous tensor:
for param in model.parameters():
    param.data = param.data.contiguous()

# 訓練
trainer.train()

# モデルの保存
question_encoder_path = "outputs_bpr/question_encoder"
model.question_encoder.save_pretrained(question_encoder_path)
tokenizer.save_pretrained(question_encoder_path)

passage_encoder_path = "outputs_bpr/passage_encoder"
model.passage_encoder.save_pretrained(passage_encoder_path)
tokenizer.save_pretrained(passage_encoder_path)
