# Disaster Response Pipeline Project
Web application which uses machine learning to categorize messages from disasters victims

### Table of Contents

1. [Installation](#installation)
2. [Training](#training)
2. [Files](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

1. Create virtual envinroment 
'''python
python -m venv *env_name*
'''

2. Install necessary libraries
'''python
pip install -r requirements.txt
'''

3. Export path into env_name/bin/activate by writing on a top of a file
'''bash
export OLD_PYTHONPATH="$PYTHONPATH"
export PYTHONPATH="path_to_project/disaster-response-project/"
'''

4. Run the following command in the app's directory to run your web app.
    `python run.py`

5. Go to http://0.0.0.0:3001/

## Training model <a name="training"></a>
If you want to re-train model run
'''python
python train_classifier.py 'path_to_database' 'classifier.pkl'
'''

If you want to run script with ETL pipeline run
'''python
python process_data.py 'disaster_messages.csv' 'disaster_categories.csv' 'DisasterResponse.db'
'''

## Files Descriptions <a name="files"></a>

| Module        | File           | Explanation  |
| ------------- |:-------------:| -----:|
| app           | run.py         | run flask web application |
| app           | templates      | html templates |
| data          | process_data.py| ETL pipeline for preparing data |
| models        | train_classifier.py| ML pipeline for training and evaluating classifier |

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The MIT License
