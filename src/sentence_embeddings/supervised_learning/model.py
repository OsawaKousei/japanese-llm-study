from unsup_model import SimCSEModel

base_model_name = "cl-tohoku/bert-base-japanese-v3"
sup_model = SimCSEModel(base_model_name=base_model_name, mlp_only_train=False)
