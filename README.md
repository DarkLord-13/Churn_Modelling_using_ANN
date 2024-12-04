# ANN Basic Classification

This project demonstrates a basic Artificial Neural Network (ANN) for classification tasks using the Churn Modelling dataset. The notebook is implemented in Python and uses libraries such as TensorFlow, NumPy, Pandas, Seaborn, and Matplotlib.

## Dataset

The dataset used in this project is the Churn Modelling dataset, which includes customer information such as Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, whether they have a credit card, whether they are active members, Estimated Salary, and whether they exited the service.

## Requirements

- Python 3
- TensorFlow
- NumPy
- Pandas
- Seaborn
- Matplotlib

## Usage

1. Import the necessary libraries:
    ```python
    import tensorflow as tf
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    ```

2. Load the dataset:
    ```python
    dataset = pd.read_csv('/content/Churn_Modelling.csv')
    dataset = dataset.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
    ```

3. Preprocess the data:
    ```python
    geography = pd.get_dummies(dataset['Geography'], drop_first=True)
    gender = pd.get_dummies(dataset['Gender'], drop_first=True)
    ```

4. Implement and train the ANN model:
    ```python
    # Define and compile the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=6, activation='relu'),
        tf.keras.layers.Dense(units=6, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=100)
    ```

5. Evaluate the model:
    ```python
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    ```

## Google Colab

You can run this notebook on Google Colab using the following link:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DarkLord-13/Machine-Learning-01/blob/main/ANN_basic(classification).ipynb)

## Author

Nishant Kumar

Feel free to fork the project and raise an issue if you have any questions or suggestions.
