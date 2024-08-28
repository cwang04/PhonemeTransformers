
MODELS=(gpt2_85M)

for model in ${MODELS[@]};
do
    echo "Launching job with model: $model with text BPE tokenizer"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_text model=$model experiment.name=$model-bpe-txt-l experiment.resume_run_id=
    echo "Launching job with model: $model with text BPE tokenizer (no word boundaries)"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_text model=$model tokenizer=babylm_text_bpe_spaceless experiment.name=$model-bpe-txt-spaceless-l experiment.resume
    echo "Launching job with model: $model with character tokenizer"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_text model=$model tokenizer=babylm_text_char experiment.name=$model-char-txt-l
    echo "Launching job with model: $model with character tokenizer (no word boundaries)"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_text model=$model tokenizer=babylm_text_char_spaceless experiment.name=$model-char-txt-spaceless-l

    echo "Launching job with model: $model with phoneme BPE tokenizer"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_phoneme model=$model tokenizer=babylm_phoneme_bpe experiment.name=$model-bpe-phoneme-l
    echo "Launching job with model: $model with phoneme BPE tokenizer (no word boundaries)"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_phoneme model=$model tokenizer=babylm_phoneme_bpe_spaceless experiment.name=$model-bpe-phoneme-spaceless-l experiment.resume_run_id=0k2nztmy
    echo "Launching job with model: $model with phoneme tokenizer"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_phoneme model=$model tokenizer=babylm_phoneme experiment.name=$model-phoneme-l experiment.resume_run_id=r9yjz5r2
    echo "Launching job with model: $model with phoneme tokenizer (no word boundaries)"
    sbatch launch_slurm.wilkes3 experiment=babylm_comparison_phoneme model=$model tokenizer=babylm_phoneme_spaceless experiment.name=$model-phoneme-spaceless-l experiment.resume_run_id=agch93rx
done