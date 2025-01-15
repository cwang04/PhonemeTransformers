#MODELS=("gpt2_600k" "gpt2_1M" "gpt2_19M" "gpt2_85M")
MODELS=("gpt2_400k" "gpt2_600k" "gpt2_800k" "gpt2_1M" "gpt2_5M" "gpt2_19M")
#DATA_SIZES=(300000 600000 800000 1300000 2000000 3000000 6000000 8000000 13000000 20000000)
DATA_SIZES=(300000 700000 1800000 3000000 7000000 18000000)

# Also do maximum data size
for model in ${MODELS[@]};
do
    echo "Launching jobs for data size: $data_size and model: $model"
    sbatch launch_slurm.wilkes3 experiment=childes_comparison_01 model=$model experiment.name=$model-full-01 $@
    sbatch launch_slurm.wilkes3 experiment=childes_comparison_03 model=$model experiment.name=$model-full-03 $@
    sbatch launch_slurm.wilkes3 experiment=childes_comparison_05 model=$model experiment.name=$model-full-05 $@
done

for data_size in ${DATA_SIZES[@]};
do
    for model in ${MODELS[@]};
    do
        echo "Launching jobs for data size: $data_size and model: $model"
        sbatch launch_slurm.wilkes3 experiment=childes_comparison_01 model=$model data_preprocessing.subsample=$data_size data_preprocessing.subsample_type=$tokens experiment.name=$model-$data_size-01 $@
        sbatch launch_slurm.wilkes3 experiment=childes_comparison_03 model=$model data_preprocessing.subsample=$data_size data_preprocessing.subsample_type=$tokens experiment.name=$model-$data_size-03 $@
        sbatch launch_slurm.wilkes3 experiment=childes_comparison_05 model=$model data_preprocessing.subsample=$data_size data_preprocessing.subsample_type=$tokens experiment.name=$model-$data_size-05 $@
    done
done


