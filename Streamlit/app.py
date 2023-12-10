import streamlit as st
import pandas as pd
import plotly.express as px

# Function to load predictions
@st.cache
def load_predictions(filename):
    data = pd.read_csv(filename)
    data['Month'] = pd.to_datetime(data['Month']).dt.strftime('%Y-%m')
    return data

# Title and Introduction
st.title('2022 Receipt Count Predictions')
st.markdown("## Explore the Receipt Count Predictions from Different Models")

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_option = st.sidebar.selectbox(
    'Choose a model:',
    ('LSTM', 'RNN', 'Prophet')
)

# Load and display the selected model's predictions
filename_map = {'LSTM': '/Users/ekanshtrivedi/Fetch-MLE/results/LSTM_predictions.csv', 
                'RNN': '/Users/ekanshtrivedi/Fetch-MLE/results/RNN_predictions.csv', 
                'Prophet': '/Users/ekanshtrivedi/Fetch-MLE/results/Prophet_predictions.csv'}

predictions = load_predictions(filename_map[model_option])

# Display Data Table
st.markdown("### Predicted Receipt Counts")
st.dataframe(predictions)

# Plotting Predictions
st.markdown("### Interactive Prediction Graph")
fig = px.line(predictions, x='Month', y='Predicted_Receipt_Count', 
              title=f'Monthly Predicted Receipt Counts for 2022 ({model_option} Model)', 
              markers=True)
st.plotly_chart(fig, use_container_width=True)

