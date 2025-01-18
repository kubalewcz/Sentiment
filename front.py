
from flask import Flask, request, render_template
from main import predict_sentiment, predict_csv
import pandas as pd


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentence = ""
    sentiment = None
    csv_results = None

    if request.method == 'POST':
        if 'analyze_text' in request.form:
            sentence = request.form['sentence']
            sentiment = predict_sentiment(sentence)

        elif 'analyze_csv' in request.form:

            if 'csv_file' in request.files:
                file = request.files['csv_file']
                data = pd.read_csv(file)
                csv_results = predict_csv(data)
                print(csv_results)

    return render_template('index.html', sentence=sentence, sentiment=sentiment, csv_results=csv_results)


if __name__ == '__main__':
    app.run(debug=True)
