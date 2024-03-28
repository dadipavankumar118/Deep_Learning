import streamlit as st
from create_data import page_1_function
from select_data import page_2_function

#st.title("TensorFlow Play Ground")

    # Create a multiselect widget to select the page
page_options = ["Home", "Create Data", "Select Data"]
selected_page = st.sidebar.radio('',page_options)

if "Home" in selected_page:
    styled_text = """<div style="font-size: 50px; color: white; text-align: left;">
    <span style="font-weight: bold;">TensorFlow Play Ground</span> 
    </div>
    <div style="font-size: 30px; color: skyblue; text-align: right;">
        <span style="font-weight: bold;">Pavankumar Dadi</span>
    </div>"""

# Display the styled text using Markdown
    st.markdown(styled_text, unsafe_allow_html=True)
    image = 'tensor.jpg' 
    st.image(image,use_column_width=True)
    

if "Create Data" in selected_page:
    page_1_function()

if "Select Data" in selected_page:
    page_2_function()