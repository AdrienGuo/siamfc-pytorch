# bin/bash

# Colors for terminal
GREEN="\e[32m"
ENDCOLOR="\e[0m"

# Settings
# model="./pretrained/official/siamfc_alexnet_e50.pth"
model="./models/CLAHE_all_all/ckpt38.pth"
part="test"
data="PatternMatch_test"  # all / PatternMatch_test
criteria="mid"  # all / mid
target="multi"  # one / multi
# official / origin / official_origin / siamfc / tri_origin / tri_127_origin
method="tri_origin"
bg="1.0"  # background

# Double Check
echo -e "${GREEN}=== Your Demo Parameters ===${ENDCOLOR}"
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
    tools/demo.py \
    --model ${model} \
    --part ${part} \
    --data ${data} \
    --data_path "./data/TRI/${part}/${data}" \
    --criteria ${criteria} \
    --target ${target} \
    --method ${method} \
    --bg ${bg}