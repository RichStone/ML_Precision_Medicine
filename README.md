## Version control installation of existing Notebook via jupytext

```python
# do this once
pip install jupytext
jupyter notebook --generate-config -y
echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"' >> ~/.jupyter/jupyter_notebook_config.py
# allow notebook to automatically generate .py code from your notebook 
jupytext --set-formats ipynb,VERSION-CONTROL-OUTPUT-DIRECTORY-NAME//py NOTEBOOK_NAME.ipynb

# git add, commit, push the .py file
# you can ignore the .ipynb file in your repository

# as a collaborator you can then git pull the repository or changes
# update your notebook with the repo's changes from the .py file
jupytext --to notebook --update VERSION-CONTROL-OUTPUT-DIRECTORY-NAME/NOTEBOOK_NAME.py
```

further reference: https://github.com/mwouts/jupytext#jupytext-commands-in-jupyterlab

## Installation of python projects and Jupyter
```python
cd project_folder

# create your virtual environment

pip install -r requirements.txt

# add kernel to notebook
python -m ipykernel install --user --name=$VENV-NAME

# choose your $VENV-NAME kernel in notebook
```
