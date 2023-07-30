import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import requests 
from PIL import Image
from streamlit_lottie import st_lottie
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential


st.set_page_config(page_title="Galaxy Shape Classification", page_icon=":sun:", layout="wide")

#st.image("E:\Galaxy_App_Test\cosmic_dust-wallpaper-1920x1080.jpg")

model=tf.keras.models.load_model("E:\\Galaxy_App_Test\\final_galaxy.h5")

def load_lottieurl(url):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
local_css("E:\Galaxy_App_Test\style.css")

#Assets
spaceman=load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_rn8hog3p.json")
rocket=load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_zsmmmni7.json")
email=load_lottieurl("https://assets6.lottiefiles.com/private_files/lf30_o0calpsv.json")


with st.container():
    left_column,right_column=st.columns((2,1))
    with left_column:
        st.title("Galaxy Shape Classification")
    with right_column:
        st_lottie(rocket,height=100,key="rocket")



with st.container():
    st.write("---")
    left_column,right_column=st.columns((1,2))
    with left_column:   
        uploaded_file=st.file_uploader("Choose a Image File",type="jpg")
        
    with right_column:
        st_lottie(spaceman,height=400,key="spaceman")

map_dict={0:'elliptical',
          1: 'irregular',
          2: 'lenticular',
          3: 'spiral'}

if uploaded_file is not None:
    file_bytes=np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image=cv2.imdecode(file_bytes,1)
    opencv_image=cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
    resized=cv2.resize(opencv_image,(150,150))
    st.image(opencv_image,channels='RGB')
    
    #resized=mobilenet_v2_preprocess_input(resized)
    img_reshape=resized[np.newaxis,...]
    
    Generate_pred=st.button("Generate Prediction")
    if Generate_pred:
        prediction=model.predict(img_reshape).argmax()
        st.title("Predicted Label for the Image is {}".format(map_dict[prediction]))
        
        
with st.container():
    st.write("---")
    st.header("Get In Touch With Us!")
    st.write("##")
    
    contact_form="""
    <form action="https://formsubmit.co/demo23galaxy@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your Name" required>
     <input type="email" name="email" placeholder="Your Email" required>
     <textarea name="message" placeholder="Your Message Here" required></textarea>
     <button type="submit">Send</button>
    </form>
    """
    
    left_column,right_column=st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st_lottie(email,height=200,key="mail")