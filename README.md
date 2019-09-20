# ki-demonstrator_wba_ima

## Setup pycharm
1. Open pycharm
2. Go to File/Settings/Project: ki-demonstrator_wba_ima/Project Interpreter
3. Set project interpreter to python.exe of conda environment.
- For Win10 usually at: C:/Users/<user>/AppData/Local/Continuum/anaconda3/envs/ki-demonstrator_wba_ima/python.exe

## Setup necessary libraries
1. enter bash and install following libraries
- Run: pip3 install flask flask_wtf wtforms sklearn numpy pandas graphviz pydotplus

## Start demonstrator
1. Run app.py
2. Click on url in console

# Serve demonstrator with publicly available URL
1. Sign up and use https://ngrok.com

# Troubleshooting

Problem:
- ImportError: DLL load failed: The specified module could not be found.
Solution:
1. Download numpy-1.16.4+mkl-cp37-cp37m-win_amd64.whl from here: https://www.lfd.uci.edu/~gohlke/pythonlibs/
2. Open console and go to location of downloaded file
4. Activate conda environment: conda activate ki-demonstrator_wba_ima
5. Run: pip install --upgrade --force-reinstall .\numpy-1.16.4+mkl-cp37-cp37m-win_amd64.whl