from flask import Flask, request, render_template
from main import predict_sentiment

# Initialize the Flask application
app = Flask(__name__)

# Define a route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve the sentence from the form
        sentence = request.form['sentence']
        # Analyze the sentiment
        sentiment = predict_sentiment(sentence)
        # Return the result
        return render_template('index.html', sentence=sentence, sentiment=sentiment)
    return render_template('index.html', sentence='', sentiment='')


# Run the application
if __name__ == '__main__':
    app.run(debug=True)
