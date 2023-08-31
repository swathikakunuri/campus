import numpy as np
from flask import Flask, render_template, request
import pickle
from sklearn.impute import SimpleImputer

app = Flask(__name__, template_folder="templates")
model = pickle.load(open('train.pkl', 'rb'))

import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "M7de4jzg0bUyyISCMC_S1oWbA45CGEQBCmwdvli_MOiY"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    ssc_p = float(request.form.get('ssc_p'))
    hsc_p = float(request.form.get('hsc_p'))
    hsc_s = int(request.form.get('hsc_s'))
    degree_p = float(request.form.get('degree_p'))
    degree_t = int(request.form.get('degree_t'))
    workex = int(request.form.get('workex'))
    etest_p = float(request.form.get('etest_p'))
    specialization = request.form.get('specialization')
    mba_p = float(request.form.get('mba_p'))

    
    arr = np.array([[ssc_p, hsc_p, hsc_s, degree_p, degree_t, workex, etest_p, specialization, mba_p]])
    brr=np.asarray(arr,dtype=float)

    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"field": [[ssc_p, hsc_p, hsc_s, degree_p, degree_t, workex, etest_p, specialization, mba_p]], "values":[[ssc_p, hsc_p, hsc_s, degree_p, degree_t, workex, etest_p, specialization, mba_p]]}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/acae7dd2-d6e1-4850-aa8c-fdffb4327237/predictions?version=2021-05-01', json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken})
    print("response_scoring")
    predictions=response_scoring.json()
    print(response_scoring.json())
    output = model.predict(arr)
    
    print("Final prediction :",output)
    print(output)
    if output == 1:
        out = 'You have high chances of getting placed!!!'
    else:
        out = 'You have low chances of getting placed. All the best.'
        
    return render_template('out1.html', output=out) 

if __name__ == '__main__':
    app.run(debug=True)




