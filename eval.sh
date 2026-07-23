MULTI_CARD=false
RANDOM_RUN_ID=$(shuf -i 100000-999999 -n 1)
RESULTS_FOLDER="test_results/"

SAMPLING_STEPS=1
BATCH_SIZE=8
TARGET_DATASET="CAMO COD10K NC4K"

LAUNCH_COMMAND="python"
if ${MULTI_CARD}; then
    RANDOM_PORT=$(shuf -i 20000-29999 -n 1)
    LAUNCH_COMMAND="accelerate launch --mixed_precision=fp16 --main_process_port ${RANDOM_PORT}"
fi

CONFIG_FILE="config/model.yaml" # Change to your config file
CHECKPOINT="checkpoints/CODiff_model.pt" # Change to your checkpoint

echo -e "\033[32m [Config file: ${CONFIG_FILE}] \033[0m"
echo -e "\033[32m [Checkpoint: ${CHECKPOINT}] \033[0m"

if [ -d "${RESULTS_FOLDER}" ]; then
    echo -e "\033[31m [Results folder ${RESULTS_FOLDER} exists, deleting...] \033[0m"
    rm -rf ${RESULTS_FOLDER}
else
    echo -e "\033[32m [Results folder ${RESULTS_FOLDER} does not exist, creating...] \033[0m"
    mkdir -p ${RESULTS_FOLDER}
fi
echo -e "\033[32m Sampling ${SAMPLING_STEPS} steps \033[0m"
${LAUNCH_COMMAND} sample.py \
    --config=${CONFIG_FILE} \
    --results_folder=${RESULTS_FOLDER} \
    --checkpoint=${CHECKPOINT} \
    --batch_size=${BATCH_SIZE} \
    --num_sample_steps=${SAMPLING_STEPS} \
    --target_dataset ${TARGET_DATASET}