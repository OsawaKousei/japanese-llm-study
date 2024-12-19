flexeval_lm \
--language_model flexeval.core.language_model.hf_lm.HuggingFaceLM  \
--language_model.model "/home/kousei/japanese-llm-study/src/preference-tuning/PT_results/checkpoint-100" \
--eval_setup "vicuna-ja" \
--eval_setup.gen_kwargs '{do_sample: true, temperature: 0.7, top_p: 0.9, max_new_tokens: 1024}' \
--save_dir "/home/kousei/japanese-llm-study/src/preference-tuning/eval_results" \
--force true
