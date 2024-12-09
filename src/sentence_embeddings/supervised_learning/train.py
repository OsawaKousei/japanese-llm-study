import logging
from logging import Formatter, StreamHandler, getLogger

from model import sup_model
from prepare import sup_train_collate_fn, sup_train_dataset, test_dataset, valid_dataset
from transformers import AutoTokenizer, TrainingArguments
from unsup_train import SimCSETrainer, compute_metrics

# ログの設定
if __name__ == "__main__":
    logger = getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
else:
    logger = getLogger("__main__").getChild(__name__)

sup_training_args = TrainingArguments(
    output_dir="outputs_sup_simcse",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=5e-5,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=250,
    logging_steps=250,
    save_steps=250,
    save_total_limit=1,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="spearman",
    remove_unused_columns=False,
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sup_trainer = SimCSETrainer(
    model=sup_model,
    args=sup_training_args,
    data_collator=sup_train_collate_fn,
    train_dataset=sup_train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

# モデルのパラメータを連続的なメモリ領域に配置
# 以下のエラーを回避するために追記
# ValueError: You are trying to save a non contiguous tensor:
for param in sup_model.parameters():
    param.data = param.data.contiguous()

sup_trainer.train()

# 評価
valid_result = sup_trainer.evaluate(valid_dataset)
test_result = sup_trainer.evaluate(test_dataset)

logger.info("valid_result: \n%s", valid_result)
logger.info("test_result: \n%s", test_result)

# モデルの保存
encoder_path = "outputs_sup_simcse/encoder"
sup_model.encoder.save_pretrained(encoder_path)
tokenizer.save_pretrained(encoder_path)
