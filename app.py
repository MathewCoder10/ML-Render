from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    data = request.get_json(force=True)
    cgpa = float(data['cgpa'])
    iq = int(data['iq'])
    profile_score = int(data['profile_score'])

    # Prediction
    result = model.predict(np.array([cgpa, iq, profile_score]).reshape(1, 3))

    if result[0] == 1:
        result = 'Placed'
    else:
        result = 'Not Placed'

    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(debug=True)
