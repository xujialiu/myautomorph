export PYTHONPATH=.:$PYTHONPATH

if [ -z "${AUTOMORPH_DATA}" ]; then
  AUTOMORPH_DATA=".."
fi

for model in 'efficientnet'
do
    for n_round in 0
    do
    seed_number=$((42-2*n_round))
    python M1_Retinal_Image_quality_EyePACS/test_outside.py --e=1 --b=64 --task_name='Retinal_quality' --model=${model} --round=${n_round} --train_on_dataset='EyePACS_quality' \
    --test_on_dataset='customised_data' --test_csv_dir="${AUTOMORPH_DATA}/Results/M0/images/" --n_class=3 --seed_num=${seed_number}

    
    done

done


