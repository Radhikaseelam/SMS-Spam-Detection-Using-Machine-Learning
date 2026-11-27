import joblib

# Load trained model
model = joblib.load("models/spam_model.pkl")

print("SMS SPAM DETECTION SYSTEM")
print("-------------------------")

while True:

    message = input("\nEnter your SMS message (or exit): ")

    if message.lower() == "exit":
        print("Program Closed.")
        break

    if not message.strip():
        print("Please enter a message.")
        continue

    result = model.predict([message])[0]

    if result == "spam":
        print("Result: SPAM MESSAGE")
    else:
        print("Result: NOT SPAM")
