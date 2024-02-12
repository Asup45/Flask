import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

with open("C:/Users/GRETA/Desktop/Exercice/Classification/Flask/Workshop2/model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')
                              
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 4)
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/predict', methods=['POST'])                                 
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        
        try:
            result = ValuePredictor(to_predict_list)
            prediction = str(result)
        except Exception as e:
            prediction = "Error in prediction: " + str(e)
        
        return render_template("predict.html", prediction=prediction)
                        
if __name__ == "__main__":
    app.run(debug=True)