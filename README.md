## Installation of each project
```
cd project_folder

# create your virtual environment

pip install -r requirements.txt

# add kernel to notebook
python -m ipykernel install --user --name=my-virtualenv-name

# choose kernel in notebook
```

## Version control installation of existing Notebook via jupytext

```
pip install jupytext

# allow notebook to automatically generate .py code from your notebook 
jupytext --set-formats ipynb,version-control-output//py NOTEBOOK_NAME.ipynb

# git add, commit, push, pull the .py file
# you can ignore the .ipynb file from now on
```
