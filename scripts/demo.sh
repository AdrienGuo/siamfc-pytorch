# bin/bash

# Color for terminal
GREEN="\e[32m"
ENDCOLOR="\e[0m"

# Settings
# model="./pretrained/official/siamfc_alexnet_e50.pth"
# model="./pretrained/ckpt100.pth"
model="./models/all_mid/ckpt100.pth"
data="./data/TRI/train/tmp"
criteria="mid"  # all / mid
target="one"  # one / multi
method="official"  # official / origin / official_origin / siamfc
bg="1.0"  # background

# Double Check
echo -e "${GREEN}=== Your Demo Parameters ===${ENDCOLOR}"
echo -e "Model: ${GREEN}${model}${ENDCOLOR}"
echo -e "Data: ${GREEN}${data}${ENDCOLOR}"
echo -e "Criteria: ${GREEN}${criteria}${ENDCOLOR}"
echo -e "Target: ${GREEN}${target}${ENDCOLOR}"
echo -e "Method: ${GREEN}${method}${ENDCOLOR}"
echo -e "Background: ${GREEN}${bg}${ENDCOLOR}"
sleep 1

# python3 script
python3 \
    tools/demo.py \
    --model ${model} \
    --data ${data} \
    --criteria ${criteria} \
    --target ${target} \
    --method ${method} \
    --bg ${bg}