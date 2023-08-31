import numpy as np
from flask import Flask, render_template, request
import pickle
from sklearn.impute import SimpleImputer

app = Flask(__name__, template_folder="templates")
model = pickle.load(open('train.pkl', 'rb'))

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
    output = model.predict(arr)
    
    if output == 1:
        out = 'You have high chances of getting placed!!!'
    else:
        out = 'You have low chances of getting placed. All the best.'
        
    return render_template('out1.html', output=out) 

if __name__ == '__main__':
    app.run(debug=True)
