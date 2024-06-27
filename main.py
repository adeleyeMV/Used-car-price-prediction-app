
import streamlit as st
from predict_module import *
from visualization_module import *

# Main function
def main():
    st.set_page_config(page_title="Car Price Prediction App", layout="wide")

    # Add navigation sidebar
    page = st.sidebar.selectbox("Select a page", ["Home", "Make prediction"])
    
    # Render the selected page
    if page == "Home":
        home_page()
    elif page == "Make prediction":
        input_page()
 
if __name__ == "__main__":
    main()