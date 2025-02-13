# Twitter-Sentiment-Analysis

## Overview
This project performs sentiment analysis on Twitter data using a Recurrent Neural Network (RNN) model. The goal is to classify tweets into positive, negative, or neutral sentiments based on the text.

## Dataset
- The dataset is sourced from Kaggle and contains tweets labeled with sentiment categories.
- It is stored in a CSV file with columns such as `tweet_id`, `text`, and `sentiment`.

## Requirements
To run this project, you need the following dependencies:

```bash
pip install numpy pandas tensorflow keras scikit-learn nltk
```

## Model Architecture
- The model uses an RNN-based approach with an embedding layer, LSTM/GRU units, and a dense output layer.
- The loss function used is `categorical_crossentropy`, and the optimizer is `adam`.

## Preprocessing Steps
1. Data cleaning: Remove special characters, URLs, and punctuation.
2. Tokenization: Convert tweets into tokens.
3. Padding: Ensure uniform input length for the model.
4. Encoding: Convert text labels into numerical format.

## Training the Model
Run the following command to train the model:

```python
python train.py
```

## Evaluation
- The trained model is evaluated using accuracy, precision, recall, and F1-score.
- Performance is visualized using confusion matrices and ROC curves.

## Usage
To use the trained model for prediction:

```python
python predict.py --text "This is an amazing day!"
```

## Results
- The model achieves an accuracy of ~X% on the test dataset.
- Example predictions:
  - "I love this product!" → Positive
  - "This was a terrible experience." → Negative

## Future Improvements
- Use transformer-based models like BERT for better accuracy.
- Expand dataset to include more diverse tweets.
- Implement real-time sentiment analysis on live Twitter streams.

## Contributors
- Kashaf Shahid

## License
This project is licensed under the Apache License 2.0.

