
from flask import Flask, request, render_template
from backend import predict_sentiment, predict_csv
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

                num_rows = int(request.form.get('num_rows', 10))
                num_rows = min(num_rows, len(data))
                df_subset = data.head(num_rows)

                data = predict_csv(df_subset, num_rows)

                csv_results = df_subset.to_dict(orient="records")


    return render_template('index.html', sentence=sentence, sentiment=sentiment, csv_results=csv_results)


if __name__ == '__main__':
    app.run(debug=True)
