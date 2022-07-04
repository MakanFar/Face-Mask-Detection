import streamlit as st
from PIL import Image
from img_classification import teachable_machine_classification

uploaded_file = st.file_uploader("Choose a png/jpg ...", type=["jpg", "png"])
st.write("""
         # Face Mask Classification
         """
         )
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("Classifying...")
    teachable_machine_classification(image, 'mask_model.h5')
    st.image('/content/output.png', caption='Classification result.', use_column_width=True)