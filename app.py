from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
import time


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel('gemini-pro-vision')

def get_gemini_response(input,image,prompt):
    response = model.generate_content([input,image[0],prompt])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")


## streamlit app section


st.title("Fashion Recommendations App")

# Dropdown for gender
gender = st.selectbox("You identify as a:", ("Male", "Female"))

size = st.selectbox("you prefer to wear clothes with size:", ("XS", "S","M","L","XL","XXL"))

# Dropdown for occasion type
occasion_type = st.selectbox("Dressing up for which occasion type:", ("Formal", "Casual", "Partywear"))

# Image upload section
uploaded_images = st.file_uploader("Upload any image of dress item you want to style: ", type=["png", "jpg", "jpeg"])

# Display uploaded images
# if uploaded_images is not None:
#     for image in uploaded_images:
#         image=Image.open(image)
#         st.image(image, caption="Dress item", use_column_width=True)
image_container = st.container()
if uploaded_images is not None:
    image=Image.open(uploaded_images)
    image_container.image(image, caption="Dress item",width=200)

# Text input box
text_input = st.text_input("Any special requests?!", key="input")

input = "Suggest a dress for {0} person with cloth size as {1} for an occasion type {2}. Include special request from user as {3}".format(gender,size,occasion_type,text_input)

input_prompt = """
You are an acclaimed fashion designer who has knowledge of world renowned fashion brands like Gucci, Armani, Louis Vuitton and Balenciaga.
You are an expert in designing dresses and suggesting trendy design ideas based on given dress item. We will provide you some images and you hae to give a complete dress idea based on the uploaded image and input in 80-100 words.
you have to provide a complete dress idea with minute details and also give accessories which can go with the dress suggestion. 
"""

submit = st.button("Generate dress suggestion")
# Submit button
if submit:
    # Process the inputs here
    image_data = input_image_details(uploaded_images)
    response = get_gemini_response(input_prompt,image_data,input)
    image_container.empty()
    st.subheader("The stylized outfit recommendation is: ")
    st.write(response)