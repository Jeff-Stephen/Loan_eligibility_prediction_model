---

# Home Loan Eligibility Prediction

## Overview
This project aims to predict the eligibility of applicants for home loans using various machine learning algorithms. The dataset used includes applicant details and loan information. The goal is to create models that can accurately predict whether an applicant is eligible for a loan.

## Project Structure
The project is structured as follows:
- `data`: Contains the dataset used for training and testing.
- `Loan_eligibility_prediction_model`: Contains the entire project including EDA, model and evaluation.
- `README.md`: Project overview and instructions.

## Dataset
The dataset consists of the following features:
- `ApplicantIncome`
- `CoapplicantIncome`
- `LoanAmount`
- `Loan_Amount_Term`
- `Credit_History`
- `Gender`
- `Married`
- `Dependents`
- `Education`
- `Self_Employed`
- `Property_Area`
- `Loan_Status` (Target variable)

## Installation
To run the project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- MLPClassifier (for neural networks)

You can install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn matplotlib tensorflow
```

## Usage
To run this project locally, follow these steps:
Clone the repository:

    ```
    git clone https://github.com/Jeff-Stephen/Loan_eligibility_prediction_model.git
    ```


## Model Performance
| Algorithm           | Accuracy |
|---------------------|----------|
| Logistic Regression | 0.78     |
| Random Forest       | 0.75     |
| Feature selection   | 0.73     |
| Neural Network      | 0.78     |


## Conclusion
Accuracy of models predicting the eligibility of persons for a home loan is compared across various algorithms.


## Acknowledgements
- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- [Scikit-learn](https://scikit-learn.org/), [TensorFlow](https://www.tensorflow.org/), and other library developers for their tools.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.


---
