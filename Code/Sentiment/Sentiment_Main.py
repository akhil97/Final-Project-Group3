import streamlit as st
import utils
import Main
import base64
import os

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set PNG as page background with shadow effect
@st.cache_data
def set_png_as_page_bg(png_file,overlay_color="#ffffff", overlay_opacity=0.7,brightness=1.0):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .bg-overlay {{
        background: {overlay_color};
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        opacity: {overlay_opacity};
    }}
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: 100% 100%;
        background-repeat: repeat;
        background-attachment: scroll;
        filter: brightness({brightness});
    }}
    </style>
    <div class="bg-overlay"></div>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

# Specify the path to your background image
background_image_path = "/Users/brundamariswamy/Downloads/court.jpg"

# Call the function to set the background image with shadow effect
set_png_as_page_bg(background_image_path)
def sentiment_main_page():
    st.markdown("<h1 style='font-size: 2 em; color: #000000; font-weight: bold;display: inline-block; text-align: right; white-space: nowrap;position: absolute; right: -150px;min-width: 600px;'> Sentiment Analysis and Entity Recognition of Legal Docuemnts</h1>", unsafe_allow_html=True)

def navigate_to_sentiment_main_page():
    sentiment_main_page()
def main():
    pages = {
        "Main Page": sentiment_main_page,
        "Next": Main.main
    }

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    # Execute the selected page's function
    pages[selected_page]()



if __name__ == "__main__":
    main()
