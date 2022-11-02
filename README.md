

 
[![0-0.png](https://i.postimg.cc/bYxR91xf/0-0.png)](https://postimg.cc/4n3tJ72W) 
[![15-0.png](https://i.postimg.cc/qMgzvTMD/15-0.png)](https://postimg.cc/rKX8f745)
[![24-0.png](https://i.postimg.cc/RFzgvd7G/24-0.png)](https://postimg.cc/1gJGHpYN)

### Result is after completing the epoches into one another (<a href="https://www.kaggle.com/code/gazu468/cyclegan-implementation-with-imbalance-dataset/notebook">Kaggle Notebook</a>)
* 1st  result after 1 epoch
* 2nd result is after completing 15 epoch
* 3rd result is completing 25 epochs



### Major Requirements

This code requires

* Python: 3.6
* Tensorflow: 2.0.0
* Keras: 2.3.1

### Preparing training and test datasets

* Download dataset from 2016, Task 3 (https://challenge.isic-archive.com/data)
* Clone this repo (obviously!)
* In this directory, make a folder in `dataset` named `isic2016` and keep all files there
* To build training set and test set

```
python data_process_isic2016.py
```

* To partition the dataset for training CycleGAN (two folders malignant and benign)

```
python data_process_gan.py
``` 

You will see that this script creates two folders `trainA` and `trainB`. Due to my utter laziness, I created `testA` and `testB` folders manually which are required for visualizing the training process of the CycleGAN. For my experiments, `testA` consisted of an image from `trainA` and vice versa.

### Training (both stages)

Now that the data is partitioned according to its class label (`trainA` -> benign and `trainB` -> malignant), train CycleGAN on this data.

* Run `train_cyclegan.ipynb`

This will result in two models: `b2m.h5` and `m2b.h5` which translate from benign -> malignant and malignant -> benign respectively. For generating the minority class (malignant) using the benign samples using the translation model:

* Run `upsampler.ipynb` to oversample and balance the dataset. (Make sure you use `b2m.h5` if you train your model. The notebook uses the pretrained weight.)

Train the classification model using the oversampled and balanced dataset

* Train classifier using `train_ISIC_2016.ipynb` 

### Evaluation

The notebook `train_ISIC_2016.ipynb` consists the code to evalute on the ISIC 2016 test set.


