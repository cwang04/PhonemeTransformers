echo "Launching job for config: Uniform"
sbatch launch_slurm.wilkes3 experiment=childes_segmentation_random_2M experiment.name=Uniform dataset.subconfig=Uniform

echo "Launching job for config: Unigram"
sbatch launch_slurm.wilkes3 experiment=childes_segmentation_random_2M experiment.name=Unigram dataset.subconfig=Unigram
