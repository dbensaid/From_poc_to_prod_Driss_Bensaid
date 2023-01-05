from predict.predict.run import TextPredictionModel
from flask import Flask, request, render_template
import os

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
parent_dir2 = os.path.dirname(parent_dir)
end_part_dir_needed= r"\train\data\artefacts\2023-01-04-21-26-31"
dir_model_needed= parent_dir2+end_part_dir_needed
print(dir_model_needed)

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['Stackoverflow_post_title']
    prediction_model = TextPredictionModel.from_artefacts(dir_model_needed)
    print(text)
    predictions = prediction_model.predict([text], top_k=1)
    return predictions

if __name__ == "__main__":
    app.run(debug=True)