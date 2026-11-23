import joblib

package = joblib.load("creditguard_model_v2.pkl")

print("Model loaded successfully!")
print("Keys:", package.keys())
print("Features:", len(package["features"]))