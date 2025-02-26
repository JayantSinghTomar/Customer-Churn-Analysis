Customer Churn Analysis
📌 Overview
Customer churn is a crucial challenge in the telecom industry. This project applies machine learning to predict customer churn and provides insights into the most important factors influencing customer retention.

📂 Project Structure
Customer-Churn-Analysis/
│── telecom_customer_churn.csv      # Dataset  
│── churn_analysis.py               # Python script for analysis  
│── requirements.txt                 # Required libraries  
│── README.md                        # Project documentation  
How to Run
1️⃣ Clone the Repository
git clone https://github.com/JayantSinghTomar/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis

2️⃣ Install Dependencies
Make sure you have Python installed. Then, install the required libraries:

pip install -r requirements.txt

3️⃣ Run the Script
Execute the Python script to perform analysis and generate results:
python churn_analysis.py

📊 Features of the Project
✅ Data Preprocessing: Handles missing values and encodes categorical data.
✅ Feature Scaling: Standardizes numerical values for better model performance.
✅ Model Training: Uses Random Forest Classifier to predict churn.
✅ Evaluation Metrics: Displays accuracy, classification report, and confusion matrix.
✅ Feature Importance: Highlights the top 10 features affecting customer churn.

📈 Results & Visualization
The script generates:
1️⃣ Confusion Matrix – Displays classification performance.
2️⃣ Feature Importance Plot – Shows the top contributing factors to churn.

Both plots are displayed and saved in the project directory.

🛠️ Technologies Used
Python 🐍
Pandas & NumPy 📊
Matplotlib & Seaborn 📉
Scikit-Learn 🤖
📩 Contributing
Feel free to fork the repo, make improvements, and submit a pull request.
