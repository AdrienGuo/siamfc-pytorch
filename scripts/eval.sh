# bin/bash

# Colors for terminal
GREEN="\e[32m"
ENDCOLOR="\e[0m"

# Settings
model="./pretrained/official/siamfc_alexnet_e50.pth"
# model="./models/all_mid/ckpt100.pth"
part="train"
data="all"
criteria="all"  # all / mid
target="multi"  # one / multi
method="origin"  # official / origin / official_origin / siamfc
bg="1.0"  # background

# Double Check
echo -e "${GREEN}=== Your Evaluate Parameters ===${ENDCOLOR}"
echo -e "Model: ${GREEN}${model}${ENDCOLOR}"
echo -e "Part: ${GREEN}${part}${ENDCOLOR}"
echo -e "Data: ${GREEN}${data}${ENDCOLOR}"
echo -e "Criteria: ${GREEN}${criteria}${ENDCOLOR}"
echo -e "Target: ${GREEN}${target}${ENDCOLOR}"
echo -e "Method: ${GREEN}${method}${ENDCOLOR}"
echo -e "Background: ${GREEN}${bg}${ENDCOLOR}"
sleep 1

# python3 script
python3 \
    ./tools/evaluate.py \
    --model ${model} \
    --part ${part} \
    --data ${data} \
    --data_path "./data/TRI/${part}/${data}" \
    --criteria ${criteria} \
    --target ${target} \
    --method ${method} \
    --bg ${bg}