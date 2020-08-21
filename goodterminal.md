phongnd@vnext.com.vn
tqd84Img
150.249.197.28

# kill process taking port
sudo lsof -t -i tcp:5000 | xargs kill -9

#list all files into a txt
ls > filenames.txt

#zip
zip -r archive_name.zip folder_to_compress

#unzip japanese
unzip -O shift-jis img.zip

#count file
ls | wc -l

#requirements
pip freeze > requirements.txt

#terraria
xset r off

#server
ssh infordio
byobu new -s ha
cd /home/vsocr/hanh


#conda new env
conda create -n myenv python

#conda env from yml file
conda env create -f environment.yml
conda activate myenv
conda env list

#activate virtual environment
python -m venv env
source env/bin/activate

#disk clean
sudo apt-get autoremove
sudo apt-get autoclean
rm -rf ~/.cache/thumbnails/*

#vue --fix
./node_modules/.bin/vue-cli-service lint --fix

