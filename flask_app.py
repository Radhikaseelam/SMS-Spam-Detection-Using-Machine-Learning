from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained machine learning model
model = joblib.load("models/spam_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""
    result_class = ""
    message = ""

    if request.method == "POST":

        message = request.form["message"].strip()

        if message:

            prediction = model.predict([message])[0]

            if prediction == "spam":
                result = "SPAM MESSAGE DETECTED!"
                result_class = "spam"

            else:
                result = "NOT SPAM - Legitimate Message"
                result_class = "not-spam"

    return render_template(
        "index.html",
        result=result,
        result_class=result_class,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)
