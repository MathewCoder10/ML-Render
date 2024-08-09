from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

# Home route to render the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route that accepts form data
@app.route('/predict', methods=['POST'])
def predict_placement():
    try:
        # Get data from the form
        cgpa = float(request.form['cgpa'])
        iq = int(request.form['iq'])
        profile_score = int(request.form['profile_score'])
        
    except (TypeError, ValueError, KeyError) as e:
        # Return to the index page with an error message
        return render_template('index.html', result='Invalid input data: ' + str(e))
    
    # Make prediction using the model
    result = model.predict(np.array([cgpa, iq, profile_score]).reshape(1, 3))

    # Interpret the prediction result
    if result[0] == 1:
        result = 'Placed'
    else:
        result = 'Not Placed'

    # Return the result to the index page
    return render_template('index.html', result=result)

# Main entry point of the application
if __name__ == "__main__":
    app.run(debug=True)
