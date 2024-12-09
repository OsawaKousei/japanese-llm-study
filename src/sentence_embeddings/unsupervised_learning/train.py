import logging
from logging import Formatter, StreamHandler, getLogger

from datasets import Dataset
from model import unsup_model
from prepare import (
    eval_collate_fn,
    test_dataset,
    unsup_train_collate_fn,
    unsup_train_dataset,
    valid_dataset,
)
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EvalPrediction, Trainer, TrainingArguments

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


def compute_metrics(pred: EvalPrediction) -> dict[str, float]:
    """評価指標の計算"""
    scores = pred.predictions
    labels, label_scores = pred.label_ids

    # スピアマン相関係数の計算
    spearman_correlation = spearmanr(scores, label_scores).statistic

    return {"spearman": spearman_correlation}


unsup_training_args = TrainingArguments(
    output_dir="outputs_unsup_simcse",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=3e-5,
    num_train_epochs=1,
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


class SimCSETrainer(Trainer):
    """SimCSEのTrainer"""

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """評価データローダーをオーバーライド"""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        return DataLoader(
            eval_dataset,
            batch_size=64,
            collate_fn=eval_collate_fn,
            pin_memory=True,
        )


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

unsup_trainer = SimCSETrainer(
    model=unsup_model,
    args=unsup_training_args,
    data_collator=unsup_train_collate_fn,
    train_dataset=unsup_train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

# モデルのパラメータを連続的なメモリ領域に配置
# 以下のエラーを回避するために追記
# ValueError: You are trying to save a non contiguous tensor:
for param in unsup_model.parameters():
    param.data = param.data.contiguous()

unsup_trainer.train()

# 評価
valid_result = unsup_trainer.evaluate(valid_dataset)
test_result = unsup_trainer.evaluate(test_dataset)

logger.info("valid_result: \n%s", valid_result)
logger.info("test_result: \n%s", test_result)

# モデルの保存
encoder_path = "outputs_unsup_simcse/encoder"
unsup_model.encoder.save_pretrained(encoder_path)
tokenizer.save_pretrained(encoder_path)
