**Retail Sales Prediction Dashboard**\n
A Streamlit web app to predict weekly sales for Walmart stores using a trained CatBoost regression model, and to visualize sales trends and feature interactions.

**Features**
•	Predict weekly sales based on inputs:

     •	Store ID
     •	Holiday Week (0 = No, 1 = Yes)
     •	Temperature
     •	Fuel Price
     •	CPI
     •	Unemployment

•	Generate future sales trend predictions for input values.
•	Upload a CSV dataset to:

     •	Visualize weekly sales trends over time
     •	Generate correlation heatmaps of numeric features

•	Time series analysis: Understand how sales change over weeks.
•	Interactive visualizations using matplotlib and seaborn.


**Installation**

1. Clone the repository:

git clone <repo_url>
cd retail-sales-prediction


2. Install dependencies:

pip install -r requirements.txt
Requirements include: streamlit, pandas, numpy, matplotlib, seaborn, joblib, catboost


**Running the App**
streamlit run app.py

•  Click Predict Weekly Sales to see the predicted sales.
•  Optionally, upload a CSV dataset to visualize sales trends.


**Input Features Explained**
| Feature      | Description                          |
| ------------ | ------------------------------------ |
| Store ID     | Unique store number                  |
| Holiday Week | 1 if week contains a holiday, else 0 |
| Temperature  | Average temperature of the week      |
| Fuel Price   | Price of fuel during the week        |
| CPI          | Consumer Price Index                 |
| Unemployment | Unemployment rate                    |

The Holiday Week feature helps the model predict spikes in sales during festival or holiday periods.


**Visualizations**
•  Predicted Sales Trend: Shows how predicted sales would change over time based on the input features.
•  Weekly Sales Trend: Plots the uploaded dataset’s sales over time.


**Saving the Model**
The model was trained using CatBoost Regressor and saved with joblib as catboost_model.joblib.

You can load this model in Python using:
import joblib
model = joblib.load("catboost_model.joblib")
