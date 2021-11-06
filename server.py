from flask import Flask, request, render_template,jsonify # Import flask libraries'
from model import load_model, make_prediction
import numpy as np
import pickle

# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder="templates")

logreg_model = load_model("logreg_model.pkl")

# Default route set as 'home'
@app.route('/')
def home():
    print("rendering")
    return render_template('home.html') # Render home.html

# Route 'classify' accepts GET request
@app.route('/classify',methods=['GET'])
def classify_type():
    print("STARTING")
    print("Attempting slen")
    sepal_len = request.args.get('slen') # Get parameters for sepal length
    print("Attempting swid")
    sepal_wid = request.args.get('swid') # Get parameters for sepal width\
    print("Attempting plen")
    petal_len = request.args.get('plen') # Get parameters for petal length
    print("Attempting pwid")
    petal_wid = request.args.get('pwid') # Get parameters for petal width

    print("Attempting prediction")
        # Get the output from the classification model
    variety = make_prediction(data=[sepal_len, sepal_wid, petal_len, petal_wid], model=logreg_model)
    print(variety)

        # Render the output in new HTML page
    return render_template('output.html', variety=variety)


# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=True) 