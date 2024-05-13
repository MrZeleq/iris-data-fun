# Iris Flowers Exploratory Data Analysis and Prediction

This repository contains exploratory data analysis (EDA) and predictive modeling code for analyzing and predicting iris flower species based on provided dataset.

## Skills Demonstrated

1. **Data Loading and Exploration**:
    - Loading the Iris dataset using sklearn.datasets.load_iris().
    - Extracting basic information about the dataset such as target names, feature names, and dataset summary statistics.

2. **Data Visualization**:
    - Visualizing the dataset using various plots like histograms, joint plots, 3D scatter plots, violin plots, box plots, pair plots, and correlation heatmaps.
    - Utilizing libraries like Matplotlib, Seaborn, and Plotly for visualization.

3. **Data Preprocessing**:
    - Converting the dataset to a Pandas DataFrame.
    - Saving the dataset to a CSV file.

4. **Model Training and Evaluation**:
    - Training multiple machine learning models including Logistic Regression, Support Vector Machine (SVM), Decision Tree Classifier, K-Nearest Neighbours (KNN), Random Forest, and AdaBoost.
    - Performing cross-validation to evaluate model performance.
    - Visualizing model evaluation metrics such as confusion matrices.

5. **Model Deployment**:
    - Saving trained models using joblib.
    - Creating a REST API for model deployment using FastAPI.
    - Defining a Pydantic model for input data.
    - Creating a POST endpoint for making predictions based on input data.

## File Description

- `data_exploration.ipynb`: Jupyter Notebook for Exploratory Data Analysis (EDA) of the Iris dataset. It includes data loading, basic information about the dataset, data visualization using seaborn and matplotlib, and correlation analysis.

- `iris_model_training.ipynb`: Jupyter Notebook for training machine learning models on the Iris dataset. It covers the training of various models such as Logistic Regression, Support Vector Machine, Decision Tree Classifier, K-Nearest Neighbors, Random Forest, and AdaBoost. It also visualizes decision trees for interpretability.

- `ml_app.py`: Python script for deploying a FastAPI-based machine learning app. It loads a pre-trained Logistic Regression model and provides an endpoint for making predictions on new data.

## Requirements

- Python 3.9, 3.10 or 3.11
- Jupyter Notebook
- Required Python libraries: Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-Learn, FastAPI

## Installation

To run the code in this project, ensure you have Python 3.9, 3.10, or 3.11 installed on your system. If not, you can download and install it from the [official Python website](https://www.python.org/downloads/).

Once Python is installed, you can set up the required libraries by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

This command will install all the necessary Python libraries listed in the `requirements.txt` file, including Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-Learn, and FastAPI.

## Usage

1. **Exploratory Data Analysis**:
    - Open `data_exploration.ipynb` in a Jupyter Notebook environment.
    - Run the code cells to explore the Iris dataset, visualize data distributions, and analyze correlations.

2. **Model Training**:
    - Open `iris_model_training.ipynb` in a Jupyter Notebook environment.
    - Run the code cells to train various machine learning models on the Iris dataset and evaluate their performance.

3. **Model Deployment**:
    - Navigate to the directory containing `ml_app.py` in your terminal.
    - Run the following command to start the FastAPI-based machine learning app:
    ```bash
    uvicorn ml_app:app --reload
    ```
    - Once the app is running, you can interact with it by sending POST requests to the defined endpoint.

<br>
Feel free to explore the code and reach out if you have any questions or suggestions!

