import numpy as np
import requests
import streamlit as st

import cv2
import os 
from os import path 
from glob import glob 
from PIL import Image 
import time 

ENDPOINT = "https://cataas.com/cat"

with st.sidebar:
    st.header("uploaded image") 
    with st.form(key="upload_image"):
        uploaded_image = None 
        input_buffer = st.file_uploader(label="charger votre image", type=("png", "jpg"))
        if input_buffer is not None:
            raw_data = input_buffer.read()
            uploaded_image = cv2.imdecode(np.frombuffer(raw_data, np.uint8), cv2.IMREAD_COLOR)
            uploaded_image = cv2.resize(uploaded_image, (512, 512))
            uploaded_image = Image.fromarray(cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB))
        

        st.form_submit_button(label="upload image")

    main_col = st.beta_columns(1)
    if uploaded_image is not None:
        main_col[0].image(uploaded_image)
    
with st.form(key="grid_reset"):
    st.header("display configuration")
    n_photos = st.slider("Number of photos:", 4, 128, 16)
    n_cols = st.number_input("Number of columns", 2, 8, 5)
    st.form_submit_button(label="Reset images and layout")


st.title("similar images")
cwd = path.dirname(path.realpath(__file__))
root = path.join(cwd, '../../mbr[SIFT]-image-similarity/source/')
contents = os.listdir(root)
cat_images = [ path.join(root, p) for p in contents ]
cat_images = cat_images[:n_photos]
cat_images = [ Image.open(p).resize((128, 128)) for p in cat_images ]

n_rows = 1 + len(cat_images) // n_cols
rows = [st.beta_container() for _ in range(n_rows)]
cols_per_row = [r.beta_columns(n_cols) for r in rows]

for image_index, cat_image in enumerate(cat_images):
    with rows[image_index // n_cols]:
        with cols_per_row[image_index // n_cols][image_index % n_cols]:
            st.image(cat_image)
        