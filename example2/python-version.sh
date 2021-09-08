#!/bin/bash

echo 'Changing python version.'
sleep '1'
touch /root/.bash_profile
echo "alias python='/usr/bin/python3'" > /root/.bash_profile
chmod 777 /root/.bash_profile
unset -f source
source /root/.bash_profile

pip3 install Pillow image
# pip install --upgrade tensorflow keras numpy pandas sklearn pillow tensorflow-gpu

echo 'Please source /root/.bash_profile manually.'
echo "Use command 'source /root/.bash_profile'"
