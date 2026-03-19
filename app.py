import json
import numpy as np
import pickle

# Load scaler only (lightweight)
scaler = pickle.load(open("scaler.pkl", "rb"))

def handler(request):
    try:
        body = json.loads(request.body)
        data = body["features"]

        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)

        # SIMPLE LOGIC (replace with your formula if needed)
        prediction = float(np.sum(data) * 50)

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": prediction})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }