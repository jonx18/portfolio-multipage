# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 08:45:45 2021

@author: Jonx-PC
"""
import fastai
from pathlib import Path
from fastai.vision.all import *
from fastai.vision.widgets import *
import wandb 
import streamlit as st
import numpy as np
import pandas as pd
from fastai.callback.wandb import *
set_seed(7)
from utils import get_unique_file
import platform

def petandfamilyidentifier_get_x(r): return path/r['fname']

def petandfamilyidentifier_get_y(r): return r['labels'].split(' ')

def show_example(image_path, container, learn):
    with container:
        img = PILImage.create(image_path)
        st.image(img, use_column_width=True)
        pred,pred_idx,probs = learn.predict(img)
        st.subheader('In this photo the model found:')
        for i in range(len(pred)):
            st.markdown(f'**{pred[i].capitalize()}** with a probability of: **{probs[pred_idx][i]*100:.00f}%**')

def app():
    if platform.system() == 'Windows':
        learn = load_learner(get_unique_file('petandfamilyidentifier_windows_v4.pkl'), cpu=True)
    else:
        learn = load_learner(get_unique_file('petandfamilyidentifier_linux_v4.pkl'), cpu=True)
    # #st.set_page_config(layout="wide")
    st.title('Pet and Family identifier V2')
    st.header('Summary')
    st.markdown('The app was conceived with the idea of auto identifier my family\'s pets:heart_eyes:, and also some family members.:sweat_smile:')
    st.markdown('This version of the app has face washing powered by Streamlit.')
    st.markdown('Finally, it includes new functionality to handle multiple images at the same time. Also a section with examples below.')
    
    uploaded_files = st.file_uploader("Upload your photos", type = ['jpg','jpeg','png'],
                                      accept_multiple_files=True)
    
    photo_container = st.beta_expander("Results",expanded = True)
        
    with photo_container:
        col1, col2 = st.beta_columns(2)
        for uploaded_file in uploaded_files:
            with st.beta_container():
                col1, col2 = st.beta_columns(2)
                bytes_data = uploaded_file.read()
                img = PILImage.create(bytes_data)
                col1.image(img, use_column_width=True)
                pred,pred_idx,probs = learn.predict(img)
                col2.subheader('In this photo the model found:')
                for i in range(len(pred)):
                    col2.markdown(f'**{pred[i].capitalize()}** with a probability of: **{probs[pred_idx][i]*100:.00f}%**')
                # col2.subheader('Is there an error? Select it please:')
                # for i in range(len(pred)):
                #     an_error = col2.checkbox(f'{pred[i].capitalize()} is not in the photo')
                #     if an_error:
    
    
    
    
    examples = st.beta_container()
    with examples:
        st.header('Examples')
        examples_containers = st.beta_columns(3)
        examples_images = [get_unique_file('new_sol.jpg'),
                           get_unique_file('new_nexus.jpeg'),
                           get_unique_file('new_nexus_jony.jpeg')]
        for example in range(len(examples_images)):
            show_example(examples_images[example], examples_containers[example], learn)





    
    
