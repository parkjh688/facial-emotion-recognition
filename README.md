# facial-emotion-recognition
Facial emotion recognition 

Using CNN Keras-tensorflow backend

### Fer2013
Fer2013 used to use **Challenges in Representation Learning: Facial Expression Recognition Challenge** in Kaggle.

> The data consists of 48x48 pixel grayscale images of faces. 
> The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. 
> The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories 
> (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

Those pictures are example of Fer2013 data.

<img src="./media/fer2013_sad.png" alt="fer2013 pic which label is sad" width="150" height="150"/>

<img src="./media/fer2013_angry.png" alt="fer2013 pic which label is angry" width="150" height="150"/>

if you want to download Fer2013 data, go [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).


### Getting Started
* Clone this repo
```
git clone https://github.com/parkjh688/facial-emotion-recognition.git
cd facial-emotion-recognition
```

* Installation
`pip install -r requirements.txt`

* train/use model
There are two ways to use this code.
1) Start with the model what I trained.
2) Start to train model on your own.

if you want `Start with the model what I trained` follow this code.

```python
# start facial emotion recognition
python real_time_fer.py
```

if you want `Start to train model on your own` follow this code.
```python
"""
your_data_path : the path where you save your data
your_model_path : the path where will you save your model
"""

# start train
python fer.py -d your_data_path -m your_model_path

# start facial emotion recognition
python real_time_fer.py -m your_model_path
```
