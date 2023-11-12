#!/bin/bash

read -p "Username:" username
read -s -p "Password:" password
echo -e "\n"

oss login -u $username -p $password

echo y | conda create -p /hy-tmp/nlp_action python=3.10

conda activate /hy-tmp/nlp_action
#!/bin/bash
pip install torch
pip install torchtext
pip install pyyaml
pip install portalocker


