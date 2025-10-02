# Iris Flower Classifier 🌸

## 📝 Overview

This project is a classic machine learning classification problem that aims to classify iris flowers into one of three species (**Setosa**, **Versicolor**, or **Virginica**) based on four feature measurements: sepal length, sepal width, petal length, and petal width.

This project demonstrates a complete, basic machine learning workflow: data loading, exploratory data analysis, model training, and evaluation. A **K-Nearest Neighbors (KNN)** classifier was used for this task.

---

## 🚀 Technologies Used

- **Language:** Python 3.x
- **Libraries:**
  - `scikit-learn` for the dataset, model, and evaluation metrics
  - `pandas` for data manipulation
  - `seaborn` & `matplotlib` for data visualization

---

## 💾 Dataset

The project uses the classic **Iris dataset**, which is included in the `scikit-learn` library. It consists of 150 samples of iris flowers, with 50 samples from each of the three species.

---

## 🛠️ How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/iris-classifier.git](https://github.com/your-username/iris-classifier.git)
    cd iris-classifier
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    (Create a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated environment).
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the script:**
    ```bash
    python iris_classifier.py
    ```
The script will first display a pairplot for visual data exploration. **You must close the plot window** for the script to continue with training and evaluation. The final results will be printed to the terminal.

---

## 📊 Results

The K-Nearest Neighbors model was evaluated on a test set (20% of the data) and achieved perfect results.

- **Accuracy:** **100%**

**Confusion Matrix:**
This matrix shows that every sample in the test set was classified correctly.
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

--- Classification Report ---
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

