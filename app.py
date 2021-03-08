import streamlit as st
from multiapp import MultiApp
from apps import home
from apps.petandfamilyidentifier import petandfamilyidentifier
from apps.petandfamilyidentifier.petandfamilyidentifier import petandfamilyidentifier_get_x, petandfamilyidentifier_get_y
from apps.dialog_recreation import dialog_recreation


app = MultiApp()


st.sidebar.markdown("""
# Examples

Use this selector to navigate through the different ML examples.
Some examples could take longer in charge because of the size of the model.

""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Pet and Family identifier", petandfamilyidentifier.app)
app.add_app("Dialog Recreation", dialog_recreation.app)
# The main app
app.run()



#This multi-page app is using the [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps) framework developed by [Praneel Nihar](https://medium.com/@u.praneel.nihar). Also check out his [Medium article](https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4).
