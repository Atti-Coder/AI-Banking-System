# =========================================================
# ADVANCED AI FRAUD DETECTION SYSTEM
# CLASSIFICATION METRICS VERSION
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report
import sqlite3
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Advanced AI Fraud Detection",
    page_icon="💳",
    layout="wide"
)
genai.configure(
    api_key="AIzaSyA_j7ZKqlUOgdHrGLPihaCkyixqDEfP3h0"
)

model = genai.GenerativeModel(
    "models/gemini-1.5-flash"
)
#=========================================================
# DATABASE SETUP
# =========================================================

conn = sqlite3.connect("banking_system.db")

cur = conn.cursor()

cur.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    balance REAL DEFAULT 0
)
''')

cur.execute('''
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    type TEXT,
    amount REAL
)
''')

conn.commit()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right,#f8fafc,#e2e8f0);
}

.main-title {
    text-align:center;
    font-size:60px;
    color:#0284c7;
    font-weight:bold;
}

.stButton>button {
    background: linear-gradient(to right,#0ea5e9,#38bdf8);
    color:white;
    border-radius:10px;
    border:none;
    font-size:18px;
    font-weight:bold;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom,#38bdf8,#0ea5e9);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOGIN SESSION
# =========================================================

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# =========================================================
# DEFAULT DATASET
# =========================================================

default_data = {

    'Amount':[100,5000,200,7000,300,10000,400,12000,150,9000],

    'Location':[1,0,1,0,1,0,1,0,1,0],

    'Time':[10,2,12,1,15,3,18,4,20,5],

    'Fraud':[0,1,0,1,0,1,0,1,0,1]
}

# =========================================================
# DATASET UPLOAD
# =========================================================

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV Dataset",
    type=['csv']
)

# =========================================================
# LOAD DATASET
# =========================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # LARGE DATASET HANDLE

    if len(df) > 5000:

        df = df.sample(5000)

    st.session_state['main_dataset'] = df

    st.sidebar.success("✅ New Dataset Loaded")

# =====================================
# USE SESSION DATASET EVERYWHERE
# =====================================

if 'main_dataset' in st.session_state:

    df = st.session_state['main_dataset']

else:

    df = pd.DataFrame(default_data)

# =========================================================
# UNIVERSAL FEATURES & TARGET
# =========================================================

# LAST COLUMN = TARGET

target_column = df.columns[-1]

# FEATURES

X = df.drop(columns=[target_column])

# TARGET

y = df[target_column]

# =====================================
# ENCODE TARGET IF NEEDED
# =====================================

if y.dtype == 'object':

    le = LabelEncoder()

    y = le.fit_transform(y)

# =====================================
# CONVERT CATEGORICAL FEATURES
# =====================================

X = pd.get_dummies(X)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# =========================================================
# MODELS
# =========================================================

lr = LogisticRegression()

dt = DecisionTreeClassifier()

rf = RandomForestClassifier()

# =========================================================
# TRAIN MODELS
# =========================================================

lr.fit(X_train,y_train)

dt.fit(X_train,y_train)

rf.fit(X_train,y_train)

# =========================================================
# PREDICTIONS
# =========================================================

lr_pred = lr.predict(X_test)

dt_pred = dt.predict(X_test)

rf_pred = rf.predict(X_test)

# =========================================================
# ACCURACY
# =========================================================

lr_acc = accuracy_score(y_test,lr_pred)

dt_acc = accuracy_score(y_test,dt_pred)

rf_acc = accuracy_score(y_test,rf_pred)

# =========================================================
# PRECISION
# =========================================================

lr_precision = precision_score(y_test,lr_pred)

dt_precision = precision_score(y_test,dt_pred)

rf_precision = precision_score(y_test,rf_pred)

# =========================================================
# RECALL
# =========================================================

lr_recall = recall_score(y_test,lr_pred)

dt_recall = recall_score(y_test,dt_pred)

rf_recall = recall_score(y_test,rf_pred)

# =========================================================
# F1 SCORE
# =========================================================

lr_f1 = f1_score(y_test,lr_pred)

dt_f1 = f1_score(y_test,dt_pred)

rf_f1 = f1_score(y_test,rf_pred)

# =========================================================
# BEST ALGORITHM
# =========================================================

accuracies = {

    "Logistic Regression": lr_acc,

    "Decision Tree": dt_acc,

    "Random Forest": rf_acc
}

best_algorithm = max(
    accuracies,
    key=accuracies.get
)

# =========================================================
# MODEL STATUS
# =========================================================

def model_status(acc):

    if acc < 0.70:

        return "❌ Underfitting"

    elif acc > 0.98:

        return "⚠ Overfitting"

    else:

        return "✅ Good Fit"

# =========================================================
# LOGIN PAGE
# =========================================================

if not st.session_state.logged_in:

    st.markdown("""
    <h1 class='main-title'>
    🔐 LOGIN TO MOVE FORWARD
    </h1>
    """, unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1563013544-824ae1b704d3?q=80&w=1400&auto=format&fit=crop",
        use_container_width=True
    )

    username = st.text_input("👤 Username")

    password = st.text_input(
        "🔑 Password",
        type="password"
    )

    if st.button("🚀 Login"):

        if username == "admin" and password == "1234":

            st.session_state.logged_in = True

            st.success("✅ Login Successful")

            st.balloons()

            st.rerun()

        else:

            st.error("❌ Invalid Username or Password")

    st.stop()

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

page = st.sidebar.radio(
    "🏦 Navigation",
    [
        "🏠 Home",
        "🤖 AI Chatbot",
        "🏦 Banking System",
        "📜 Transactions",
        "🔍 Fraud Prediction",
        "📊 Dashboard",
        "🤖 Algorithms",
        "📈 Analytics",
        "📋 Classification Metrics",
        "📄 AI Report",
        "📂 Dataset",
        "🌍 Universal AI",
        "ℹ About Project",
    ]
)

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.markdown("""
    <h1 class='main-title'>
    💳 AI Fraud Detection Dashboard
    </h1>
    """, unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1556740749-887f6717d7e4?q=80&w=1400&auto=format&fit=crop",
        use_container_width=True
    )

    st.success("🚀 Advanced Banking AI System")

# =========================================================
# DASHBOARD PAGE
# =========================================================

elif page == "📊 Dashboard":

    st.title("📊 Banking Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "💰 Total Transactions",
            len(df)
        )

    with col2:

        st.metric(
            "⚠ Fraud Transactions",
            int(df[target_column].sum())
        )

    with col3:

        st.metric(
            "✅ Genuine Transactions",
            int(df[target_column].sum())
        )

# =========================================================
# FRAUD PREDICTION PAGE
# =========================================================

elif page == "🔍 Fraud Prediction":

    st.title("🔍 Fraud Prediction")

    amount = st.number_input("💰 Transaction Amount")

    location = st.selectbox(
        "🌍 Location",
        [0,1]
    )

    time = st.slider(
        "⏰ Transaction Time",
        0,
        23
    )

    if st.button("🚀 Predict Transaction"):

        user_data = pd.DataFrame(
            [[amount,location,time]],
            columns=['Amount','Location','Time']
        )

        prediction = rf.predict(user_data)

        confidence = round(
            max(rf.predict_proba(user_data)[0])*100,
            2
        )

        if prediction[0] == 1:

            st.error("⚠ Fraudulent Transaction Detected")

            st.warning("🚨 HIGH RISK TRANSACTION ALERT 🚨")

            st.toast("🚨 FRAUD ALERT DETECTED 🚨")

        else:

            st.success("✅ Genuine Transaction")

        st.info(
            f"🤖 Confidence Score: {confidence}%"
        )

# =========================================================
# ALGORITHMS PAGE
# =========================================================

elif page == "🤖 Algorithms":

    st.title("🤖 AI Algorithms")

    st.subheader("🧠 Logistic Regression")

    st.write("""
    Used for binary classification problems.
    """)

    st.write(f"Accuracy: {lr_acc:.2f}")

    st.write(model_status(lr_acc))

    st.subheader("🌳 Decision Tree")

    st.write("""
    Works like a flowchart using conditions.
    """)

    st.write(f"Accuracy: {dt_acc:.2f}")

    st.write(model_status(dt_acc))

    st.subheader("🌲 Random Forest")

    st.write("""
    Combines multiple trees for better accuracy.
    """)

    st.write(f"Accuracy: {rf_acc:.2f}")

    st.write(model_status(rf_acc))

    st.success(
        f"🏆 Best Algorithm: {best_algorithm}"
    )

# =========================================================
# ANALYTICS PAGE
# =========================================================

elif page == "📈 Analytics":

    st.title("📈 Advanced Analytics")

    # ROC CURVE

    lr_prob = lr.predict_proba(X_test)[:,1]

    dt_prob = dt.predict_proba(X_test)[:,1]

    rf_prob = rf.predict_proba(X_test)[:,1]

    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)

    dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_prob)

    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

    lr_auc = auc(lr_fpr, lr_tpr)

    dt_auc = auc(dt_fpr, dt_tpr)

    rf_auc = auc(rf_fpr, rf_tpr)

    fig, ax = plt.subplots()

    ax.plot(
        lr_fpr,
        lr_tpr,
        label=f"Logistic Regression ({lr_auc:.2f})"
    )

    ax.plot(
        dt_fpr,
        dt_tpr,
        label=f"Decision Tree ({dt_auc:.2f})"
    )

    ax.plot(
        rf_fpr,
        rf_tpr,
        label=f"Random Forest ({rf_auc:.2f})"
    )

    ax.legend()

    ax.set_title("ROC Curve")

    st.pyplot(fig)

    # PIE CHART

    st.subheader("🥧 Fraud vs Genuine")

    fraud_count = int(df[target_column].sum())

    genuine_count = int(
        len(df)-fraud_count
    )

    fig2, ax2 = plt.subplots()

    ax2.pie(
        [fraud_count,genuine_count],
        labels=['Fraud','Genuine'],
        autopct='%1.1f%%'
    )

    st.pyplot(fig2)

# =========================================================
# CLASSIFICATION METRICS PAGE
# =========================================================

elif page == "📋 Classification Metrics":

    st.title("📋 Classification Metrics")

    metrics_df = pd.DataFrame({

        'Algorithm':[
            'Logistic Regression',
            'Decision Tree',
            'Random Forest'
        ],

        'Precision':[
            lr_precision,
            dt_precision,
            rf_precision
        ],

        'Recall':[
            lr_recall,
            dt_recall,
            rf_recall
        ],

        'F1 Score':[
            lr_f1,
            dt_f1,
            rf_f1
        ]
    })

    st.dataframe(metrics_df)

    # =====================================================
    # PRECISION CHART
    # =====================================================

    fig3, ax3 = plt.subplots()

    ax3.bar(
        metrics_df['Algorithm'],
        metrics_df['Precision']
    )

    ax3.set_title("Precision Comparison")

    st.pyplot(fig3)

    # =====================================================
    # RECALL CHART
    # =====================================================

    fig4, ax4 = plt.subplots()

    ax4.bar(
        metrics_df['Algorithm'],
        metrics_df['Recall']
    )

    ax4.set_title("Recall Comparison")

    st.pyplot(fig4)

    # =====================================================
    # F1 SCORE CHART
    # =====================================================

    fig5, ax5 = plt.subplots()

    ax5.bar(
        metrics_df['Algorithm'],
        metrics_df['F1 Score']
    )

    ax5.set_title("F1 Score Comparison")

    st.pyplot(fig5)

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    st.subheader("📄 Random Forest Classification Report")

    st.text(
        classification_report(
            y_test,
            rf_pred
        )
    )

# =========================================================
# AI REPORT PAGE
# =========================================================

elif page == "📄 AI Report":

    st.title("📄 AI Generated Report")

    st.write("""
    ## 💳 Fraud Detection Final Report
    """)

    st.write(f"🧠 Logistic Regression Accuracy: {lr_acc:.2f}")

    st.write(f"🌳 Decision Tree Accuracy: {dt_acc:.2f}")

    st.write(f"🌲 Random Forest Accuracy: {rf_acc:.2f}")

    st.write(f"🧠 Logistic Precision: {lr_precision:.2f}")

    st.write(f"🌳 Decision Tree Precision: {dt_precision:.2f}")

    st.write(f"🌲 Random Forest Precision: {rf_precision:.2f}")

    st.success(
        f"🏆 Best Algorithm: {best_algorithm}"
    )

# =========================================================
# DATASET PAGE
# =========================================================

elif page == "📂 Dataset":

    st.title("📂 Dataset Viewer")

    st.dataframe(df)
    


pass

    # =========================================================
    # BANKING SYSTEM
    # =========================================================

if page == "🏦 Banking System":

        st.title("🏦 Banking System")

        conn = sqlite3.connect(
            "banking_system.db",
            check_same_thread=False
        )

        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            balance REAL DEFAULT 0
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            type TEXT,
            amount REAL
        )
        """)

        conn.commit()

        option = st.selectbox(
            "Select Option",
            [
                "Create User",
                "Deposit",
                "Withdraw",
                "Check Balance"
            ]
        )

        username = st.text_input(
            "Enter Username"
        )

        # CREATE USER

        if option == "Create User":

            if st.button("Create User"):

                try:

                    cur.execute(
                        "INSERT INTO users(username,balance) VALUES (?,?)",
                        (username,0)
                    )

                    conn.commit()

                    st.success(
                        "✅ User Created Successfully"
                    )

                except:

                    st.error(
                        "❌ User Already Exists"
                    )

        # DEPOSIT

        elif option == "Deposit":

            amount = st.number_input(
                "Deposit Amount",
                min_value=0.0
            )

            if st.button("Deposit Money"):

                cur.execute(
                    "UPDATE users SET balance = balance + ? WHERE username=?",
                    (amount,username)
                )

                cur.execute(
                    "INSERT INTO transactions(username,type,amount) VALUES (?,?,?)",
                    (username,"Deposit",amount)
                )

                conn.commit()

                st.success(
                    "✅ Money Deposited Successfully"
                )

        # WITHDRAW

        elif option == "Withdraw":

            amount = st.number_input(
                "Withdraw Amount",
                min_value=0.0
            )

            if st.button("Withdraw Money"):

                cur.execute(
                    "SELECT balance FROM users WHERE username=?",
                    (username,)
                )

                result = cur.fetchone()

                if result:

                    balance = result[0]

                    if balance >= amount:

                        cur.execute(
                            "UPDATE users SET balance = balance - ? WHERE username=?",
                            (amount,username)
                        )

                        cur.execute(
                            "INSERT INTO transactions(username,type,amount) VALUES (?,?,?)",
                            (username,"Withdraw",amount)
                        )

                        conn.commit()

                        st.success(
                            "✅ Withdrawal Successful"
                        )

                    else:

                        st.error(
                            "❌ Insufficient Balance"
                        )

                else:

                    st.error(
                        "❌ User Not Found"
                    )

        # CHECK BALANCE

        elif option == "Check Balance":

            if st.button("Check Balance"):

                cur.execute(
                    "SELECT balance FROM users WHERE username=?",
                    (username,)
                )

                result = cur.fetchone()

                if result:

                    st.success(
                        f"💰 Current Balance: {result[0]}"
                    )

                else:

                    st.error(
                        "❌ User Not Found"
                    )

        conn.close()
                        # =========================================================
    # TRANSACTIONS
    # =========================================================

if page == "📜 Transactions":

        st.title("📜 Transaction History")

        cur.execute(
            "SELECT * FROM transactions"
        )

        data = cur.fetchall()

        transaction_df = pd.DataFrame(
            data,
            columns=['ID','Username','Type','Amount']
        )

        st.dataframe(transaction_df)

# =========================================================
# UNIVERSAL AI PAGE
# =========================================================

if page == "🌍 Universal AI":

    st.title("🌍 Universal AI System")

    uploaded_file2 = st.file_uploader(
        "Upload Any CSV Dataset",
        type=['csv']
    )

    if uploaded_file2 is not None:

        df2 = pd.read_csv(uploaded_file2)

        df2 = df2.sample(5000)

        st.session_state['main_dataset'] = df2

        df = st.session_state['main_dataset']

        st.success("✅ Dataset Loaded Successfully")

        st.subheader("📂 Dataset Preview")

        st.dataframe(df2.head())

        # =====================================
        # TARGET COLUMN
        # =====================================

        target = st.selectbox(
            "🎯 Select Target Column",
            df2.columns
        )

        # =====================================
        # TRAIN BUTTON
        # =====================================

        if st.button("🚀 Train Universal Model"):

            # FEATURES & TARGET

            X = df2.drop(columns=[target])

            y = df2[target]

            # =================================
            # ENCODE TARGET
            # =================================

            if y.dtype == 'object':

                le = LabelEncoder()

                y = le.fit_transform(y)

            # =================================
            # HANDLE CATEGORICAL FEATURES
            # =================================

            X = pd.get_dummies(X)

            # =================================
            # TASK DETECTION
            # =================================

            if len(np.unique(y)) < 20:

                task = "classification"

            else:

                task = "regression"

            st.info(f"🧠 Detected Task: {task}")

            # =================================
            # TRAIN TEST SPLIT
            # =================================

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            # =================================
            # MODEL
            # =================================

            if task == "classification":

                model = RandomForestClassifier()

            else:

                model = RandomForestRegressor()

            # =================================
            # TRAIN MODEL
            # =================================

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            st.success("✅ Model Trained Successfully")

            # =================================
            # CLASSIFICATION
            # =================================

            if task == "classification":

                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    average='weighted'
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    average='weighted'
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average='weighted'
                )

                st.subheader("📋 Classification Results")

                st.write(f"Accuracy : {accuracy:.2f}")

                st.write(f"Precision: {precision:.2f}")

                st.write(f"Recall   : {recall:.2f}")

                st.write(f"F1 Score : {f1:.2f}")

            # =================================
            # REGRESSION
            # =================================

            else:

                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                )

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                r2 = r2_score(
                    y_test,
                    predictions
                )

                st.subheader("📋 Regression Results")

                st.write(f"RMSE: {rmse:.2f}")

                st.write(f"MAE : {mae:.2f}")

                st.write(f"R2  : {r2:.2f}")
                # =====================================
# =====================================
# AI CHATBOT
# =====================================

if page == "🤖 AI Chatbot":

    st.title("🤖 AI Banking Assistant")

    st.write(
        "Talk with your AI Assistant"
    )

    # USER INPUT

    user_question = st.text_input(
        "Type Your Message"
    )

    # SEND BUTTON

    if st.button("Send"):

        if user_question != "":

            question = user_question.lower()

            # GREETINGS

            if "hi" in question or "hello" in question:

                st.success(
                    "👋 Hello! How are you today?"
                )

            elif "how are you" in question:

                st.success(
                    "😊 I'm doing great! Thanks for asking."
                )

            # BANKING

            elif "balance" in question:

                st.success(
                    "💰 Your banking balance system is active."
                )

            elif "deposit" in question:

                st.success(
                    "🏦 Deposit feature is available in Banking System."
                )

            elif "withdraw" in question:

                st.success(
                    "💸 Withdraw feature is active."
                )

            # FRAUD

            elif "fraud" in question:

                st.success(
                    "🔍 Fraud Detection AI is working successfully."
                )

            # AI

            elif "ai" in question:

                st.success(
                    "🤖 This project uses Artificial Intelligence and Machine Learning."
                )

            # DEFAULT

            else:

                st.success(
                    f"🤖 You said: {user_question}"
                )

                st.write(
                    "I'm your friendly AI Banking Assistant 😎"
                )

# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "ℹ About Project":

    st.title("ℹ About Project")

    st.write("""

# 💳 AI Banking Fraud Detection Dashboard

This project uses Artificial Intelligence and
Machine Learning algorithms to detect fraudulent
banking transactions.

# 🤖 Algorithms Used

## 🧠 Logistic Regression
Used for binary classification.

## 🌳 Decision Tree
Works like a flowchart.

## 🌲 Random Forest
Combines multiple trees for better predictions.

# 📌 Why These Algorithms?

Fraud detection is a classification problem:
- Fraud = 1
- Genuine = 0

Therefore classification algorithms are best.

# 📊 Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-Learn
- Matplotlib

# 🚀 Project Goal

To build a professional AI-powered banking
fraud detection system.

""")














