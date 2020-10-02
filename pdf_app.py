from custom_functions import *
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    path = [str(x) for x in request.form.values()][0]

    output = flask_test(path)

    return render_template('index.html', metadata_text='This is the extracted information $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    # app.run(host='0.0.0.0')