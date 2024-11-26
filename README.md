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
1. '--dataset': The data you choosed to train the model.
2. '--dev': The device you choose to run your code.
3. '--epochs': The number of epochs for ae. (Default: 8000)
4. '--memlen': The size of memory.
5. '--win_size': The size of local window.
6. '--dim': The dimension of encoder_output.
7. '--b_dim': The dimension of adapter_mapping.

## Bash
# python memstream-knn_adapter_test.py --dataset KDD --lr 1e-2 --memlen 512 --win_size 50 --dim 32 --b_dim 128 --gamma 0
# python memstream-knn_adapter_test.py --dataset NSL --lr 1e-3 --memlen 2048 --win_size 50 --dim 5 --b_dim 32 --gamma 0.5
# python memstream-knn_adapter_test.py --dataset DOS --lr 1e-2 --memlen 1024 --win_size 150 --dim 1 --b_dim 64 --gamma 0.5
# python memstream-knn_adapter_test.py --dataset ionosphere --lr 1e-2 --memlen 64 --win_size 2 --dim 9 --b_dim 16 --gamma 0.5
# python memstream-knn_adapter_test.py --dataset cardio --lr 1e-3 --memlen 64 --win_size 20 --dim 2 --b_dim 64 --gamma 1
python memstream-knn_adapter_test.py --dataset statlog --lr 1e-3 --memlen 32 --win_size 2 --dim 5 --b_dim 64 --gamma 1
# python memstream-knn_adapter_test.py --dataset satimage-2 --lr 1e-3 --memlen 64 --win_size 10 --dim 5 --b_dim 32 --gamma 0
# python memstream-knn_adapter_test.py --dataset mammography --lr 1e-2 --memlen 512 --win_size 100 --dim 1 --b_dim 64 --gamma 0.25
# python memstream-knn_adapter_test.py --dataset pima --lr 1e-2 --memlen 16 --win_size 5 --dim 2 --b_dim 128 --gamma 1
# python memstream-knn_adapter_test.py --dataset cover --lr 1e-3 --memlen 2048 --win_size 50 --dim 0.25 --b_dim 32 --gamma 1
