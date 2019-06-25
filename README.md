# ki-demonstrator_wba_ima

# Setup
1. Install anaconda: https://www.anaconda.com/
2. import environment.yml: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
3. Install pycharm: https://www.jetbrains.com/pycharm/

## Setup pycharm
1. Open pycharm
2. Go to File/Settings/Project: ki-demonstrator_wba_ima/Project Interpreter
3. Set project interpreter to python.exe of conda environment.
- For Win10 usually at: C:/Users/<user>/AppData/Local/Continuum/anaconda3/envs/ki-demonstrator_wba_ima/python.exe

## Start demonstrator
1. Run app.py
2. Click on url in console

# Troubleshooting

Problem:
- ImportError: DLL load failed: The specified module could not be found.
Solution:
1. Download numpy-1.16.4+mkl-cp37-cp37m-win_amd64.whl from here: https://www.lfd.uci.edu/~gohlke/pythonlibs/
2. Open console and go to location of downloaded file
4. Activate conda environment: conda activate ki-demonstrator_wba_ima
5. Run: pip install --upgrade --force-reinstall .\numpy-1.16.4+mkl-cp37-cp37m-win_amd64.whl