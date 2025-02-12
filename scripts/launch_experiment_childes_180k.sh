LANGUAGES=("English" "EnglishUK" "German" "Japanese" "Indonesian" "French" "Spanish" "Mandarin" "Dutch" "Polish" "Serbian" "Estonian" "Welsh" "Cantonese" "Swedish" "PortuguesePt" "Italian" "Croatian" "Catalan" "Icelandic")

for language in ${LANGUAGES[@]};
do
    echo "Launching job for language: $language"
    sbatch launch_slurm.wilkes3 experiment=childes_multilingual_180k experiment.name=$language dataset.subconfig=$language tokenizer.name=phonemetransformers/CHILDES-$language-phoneme-tokenizer
done
