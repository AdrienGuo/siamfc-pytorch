# bin/bash

# Color for terminal
GREEN="\e[32m"
ENDCOLOR="\e[0m"

# Settings
data="./data/TRI/train/tmp"
test_data="./data/TRI/test/tmp"
criteria="mid"  # all / mid
target="one"  # one / multi
method="siamfc"  # siamcar / origin / official_origin / siamfc
bg="1.0"  # background

# Double Check
echo -e "${GREEN}=== Your Train Parameters ===${ENDCOLOR}"
echo -e "Train data: ${GREEN}${data}${ENDCOLOR}"
echo -e "Test tata: ${GREEN}${test_data}${ENDCOLOR}"
echo -e "Criteria: ${GREEN}${criteria}${ENDCOLOR}"
echo -e "Target: ${GREEN}${target}${ENDCOLOR}"
echo -e "Method: ${GREEN}${method}${ENDCOLOR}"
echo -e "Background: ${GREEN}${bg}${ENDCOLOR}"
sleep 1

# python3 script
python3 \
    tools/train.py \
    --data ${data} \
    --test_data ${test_data} \
    --criteria ${criteria} \
    --target ${target} \
    --method ${method} \
    --bg ${bg}