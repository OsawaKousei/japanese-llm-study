import logging
from logging import Formatter, StreamHandler, getLogger

from datasets import Dataset
from prepare import eval_collate_fn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformers import EvalPrediction, Trainer, TrainingArguments

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
