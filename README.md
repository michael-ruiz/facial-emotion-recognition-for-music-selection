# Facial Expression Recognition for Mood-Based Song Selection

## Getting Started

### Install dependencies:
**Windows** 
```
pip install tensorflow
pip install numpy
pip install opencv-python
pip install flask
```

**Mac**
```
pip3 install tensorflow
pip3 install numpy
pip3 install opencv-python
pip3 install flask
```


### Make music directories (empty):

**Windows** 
```
mkdir frontend\static\music\angry frontend\static\music\disgust frontend\static\music\fear frontend\static\music\happy frontend\static\music\neutral frontend\static\music\sad frontend\static\music\surprise
```

**Mac**
```
mkdir -p frontend/static/music/{angry,disgust,fear,happy,neutral,sad,surprise}
```

### Run app (localhost:5000):

**Windows** 
```
cd frontend
python app.py
```

**Mac**
```
cd frontend
python3 app.py
```

## Link to dataset
https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data