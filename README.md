# AdpStream

## Datasets
You can download the processed datasets from [here](https://drive.google.com/file/d/1JNrhOr8U3Nqef1hBOqvHQPzBNWzDOFdl/view). After downloading, please unzip the files and place them into the "data" folder of the repository.
1. [KDDCUP99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
2. [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
3. [CICIDS-DoS](https://www.unb.ca/cic/datasets/ids-2018.html)
4. [Ionosphere](https://archive.ics.uci.edu/ml/index.php)
5. [Cardiotocography](https://archive.ics.uci.edu/ml/index.php)
6. [Statlog Landsat Satellite](https://archive.ics.uci.edu/ml/index.php)
7. [Satimage-2](http://odds.cs.stonybrook.edu)
8. [Mammography](http://odds.cs.stonybrook.edu)
9. [Pima Indians Diabetes](https://archive.ics.uci.edu/ml/index.php)
10. [Covertype](https://archive.ics.uci.edu/ml/index.php)

## Params
'--dataset': The data you choosed to train the model.  
'--dev': The device you choose to run your code.  
'--epochs': The number of epochs for ae. (Default: 8000)  
'--memlen': The size of memory.  
'--win_size': The size of local window.  
'--dim': The dimension of encoder_output.  
'--b_dim': The dimension of adapter_mapping.  

## Requirements
Before running the script, you need to install the dependencies listed in the requirements.txt file:  
pip3 install -r requirements.txt

## Bash
Taking dataset Cardio as an example:  
pythonAdpStream.py --dataset cardio --lr 1e-3 --memlen 64 --win_size 20 --dim 2 --b_dim 64 --gamma 1

