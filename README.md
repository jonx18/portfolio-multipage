# Portfolio multipage

# Reproducing this web app
To recreate this web app on your own computer, do the following.

### Create conda environment
Firstly, we will create a conda environment called *portfolio_multipage*
```
conda create -n portfolio_multipage python=3.7.9
```
Secondly, we will login to the *multipage* environement
```
conda activate portfolio_multipage
```
### Download and unzip this repository

Download [this repo](https://github.com/jonx18/portfolio-multipage/archive/master.zip) and unzip as your working directory.

### Install prerequisite libraries

Pip install libraries from the repository
```
pip install -r requirements.txt
```


###  Launch the app

```
streamlit run app.py
```
