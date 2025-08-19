# define your job name

date
                                                
dataset_name='ALL-SIX'

seed_number=$((42-2*seed))

job_name=20210630_uniform_thres40_${dataset_name}

python M2_Vessel_seg/test_outside_integrated.py --epochs=1 \
                                                --batchsize=8 \
                                                --learning_rate=2e-4 \
                                                --validation_ratio=10.0 \
                                                --alpha=0.08 \
                                                --beta=1.1 \
                                                --gamma=0.5\
                                                --dataset=${dataset_name} \
                                                --dataset_test=${dataset_name} \
                                                --uniform='True' \
                                                --jn=${job_name} \
                                                --worker_num=2 \
                                                --save_model='best' \
                                                --train_test_mode='test' \
                                                --pre_threshold=40.0 \
                                                --seed_num=${seed_number} \
                                                --out_test="${AUTOMORPH_DATA}/Results/M2/binary_vessel/"
                                                
                                        

date