import streamlit as st
import pandas as pd
import plotly.express as px

# Function to load predictions
@st.cache_data
def load_predictions(filename):
    data = pd.read_csv(filename)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m')
    return data

# Title and Introduction with icons
st.title('💹 2022 Receipt Count Predictions')
st.markdown("## Explore the Receipt Count Predictions for the year 2022 📈")

# Sidebar for model selection
st.sidebar.title("🔍 Model Selection")
model_option = st.sidebar.selectbox(
    'Choose a model:',
    ('LSTM', 'RNN', 'Prophet')
)

# Button to load predictions
if st.sidebar.button('Get Predictions'):
    filename_map = {'LSTM': '/Users/ekanshtrivedi/Fetch-MLE/results/LSTM_predictions-final.csv', 
                    'RNN': '/Users/ekanshtrivedi/Fetch-MLE/results/rnn_predictions-final.csv', 
                    'Prophet': '/Users/ekanshtrivedi/Fetch-MLE/results/Prophet_predictions.csv'}

    predictions = load_predictions(filename_map[model_option])

    # Display Data Table
    st.markdown("### Predicted Receipt Counts")
    st.dataframe(predictions)

    # Plotting Predictions and fixing the error
    st.markdown("### Interactive Prediction Graph")
    fig = px.line(predictions, x='Date', y='Prediction', 
                  title=f'Monthly Predicted Receipt Counts for 2022 ({model_option} Model)', 
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # Download links
    st.markdown('## Download Results')
    st.download_button(
        label="Download Data as CSV",
        data=predictions.to_csv(index=False).encode('utf-8'),
        file_name='predictions.csv',
        mime='text/csv',
    )

    st.download_button(
        label="Download Plot as Image",
        data=fig.to_image(format="png"),
        file_name='predictions_plot.png',
        mime='image/png',
    )
