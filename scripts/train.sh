# bin/bash

# Color for terminal
GREEN="\e[32m"
ENDCOLOR="\e[0m"

# Train Settings
data="all"  # all / tmp
criteria="all"  # all / mid
method="siamcar"  # siamcar / origin / official_origin / siamfc
bg="All"  # background
# Test Settings
test_data="all"  # all / tmp
# Evaluate Settings
eval_criteria="mid"  # all / mid
eval_method="origin"  # siamcar / origin / official_origin / siamfc
eval_bg="1.0"
# Others Settings
target="multi"  # one / multi

# Double Check
echo -e "${GREEN}=== Your Train Parameters ===${ENDCOLOR}"
echo -e "Train Data: ${GREEN}${data}${ENDCOLOR}"
echo -e "Train Criteria: ${GREEN}${criteria}${ENDCOLOR}"
echo -e "Train Method: ${GREEN}${method}${ENDCOLOR}"
echo -e "Background: ${GREEN}${bg}${ENDCOLOR}"
echo -e "Test Data: ${GREEN}${test_data}${ENDCOLOR}"
echo -e "Eval Criteria: ${GREEN}${eval_criteria}${ENDCOLOR}"
echo -e "Eval Method: ${GREEN}${eval_method}${ENDCOLOR}"
echo -e "Eval Background: ${GREEN}${eval_bg}${ENDCOLOR}"
echo -e "Target: ${GREEN}${target}${ENDCOLOR}"
sleep 1

# python3 script
python3 \
    tools/train.py \
    --data "./data/TRI/train/${data}" \
    --criteria ${criteria} \
    --method ${method} \
    --bg ${bg} \
    --test_data "./data/TRI/test/${test_data}" \
    --eval_criteria ${eval_criteria} \
    --eval_method ${eval_method} \
    --eval_bg ${eval_bg} \
    --target ${target} \