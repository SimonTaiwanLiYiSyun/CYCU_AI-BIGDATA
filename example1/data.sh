#!/bin/bash

export fileid=1dbcWabr3Xrr4JvuG0VxTiweGzHn-YYvW
export filename=v_data.zip

wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

unzip v_data.zip
cp -r v_data/test/ .
cp -r v_data/train/ .

rm confirm.txt && rm cookies.txt
rm v_data.zip

echo 'Changing python version.'
sleep '1'
touch /root/.bash_profile
echo "alias python='/usr/bin/python3'" > /root/.bash_profile
chmod 777 /root/.bash_profile
unset -f source
source /root/.bash_profile

# pip install pillow graphViz pydot==1.2.4
# pip3 install Pillow image graphZip pydot==1.2.4
pip3 install Pillow image
pip3 install --upgrade tensorflow keras numpy pandas sklearn pillow
# pip install --upgrade tensorflow keras numpy pandas sklearn pillow

echo 'Please source /root/.bash_profile manually.'
echo "Use command 'source /root/.bash_profile'"
