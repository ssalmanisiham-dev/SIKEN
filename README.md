# AI-Based Network Anomaly Detection SIKEN (Smart Intelligent Knowledge for Event Notification)

## About the project

This project was developed from scratch as part of an AI and cybersecurity project. The goal is to detect anomalies in network traffic using different machine learning models, display the results on a Streamlit dashboard, and send email alerts when suspicious activity is detected.

## Models used

The project includes four machine learning models:

* Random Forest
* K-Means
* Isolation Forest
* Gradient Boosting

## Dataset

The project uses the **CICIDS** dataset.

Before running the project, create a folder called **data** and place the dataset inside it.

## How to run

First, install the required libraries:

```
pip install -r requirements.txt
```

Then start the dashboard with:

```
streamlit run dashboard.py
```

## Email alerts

To enable email notifications, edit the email configuration in the code and add:

* Sender email
* Sender password (or App Password)
* Receiver email

Once configured, the application will automatically send an email when an anomaly is detected.

## Technologies

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy


## Notes

This project is intended for educational purposes and demonstrates how machine learning can be used for network anomaly detection.


*** Special thanks to my teammate Kenza for surviving this project with me. We fixed bugs, questioned our life choices, and somehow made it work. 😂 ***
