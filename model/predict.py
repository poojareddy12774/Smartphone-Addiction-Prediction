import joblib
import numpy as np
import time

model = joblib.load("stacking_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Model Loaded Successfully\n")

print("Gender Mapping:")
print("Female=0, Male=1, Other=2\n")

print("Phone Usage Purpose Mapping:")
print("Browsing=0, Education=1, Gaming=2, Other=3, Social Media=4\n")

age = int(input("Enter Age: "))
gender = int(input("Enter Gender (0=Female,1=Male,2=Other): "))

daily_usage = float(input("Daily Usage Hours: "))
sleep_hours = float(input("Sleep Hours: "))
social_interactions = int(input("Social Interactions: "))
exercise_hours = float(input("Exercise Hours: "))
anxiety = int(input("Anxiety Level: "))
depression = int(input("Depression Level: "))
self_esteem = int(input("Self Esteem: "))
parental_control = int(input("Parental Control Level: "))
screen_before_bed = float(input("Screen Time Before Bed: "))
phone_checks = int(input("Phone Checks Per Day: "))
apps_used = int(input("Apps Used Daily: "))
time_social = float(input("Time on Social Media: "))
time_gaming = float(input("Time on Gaming: "))
time_education = float(input("Time on Education: "))

purpose = int(input("Phone Usage Purpose (0=Browsing,1=Education,2=Gaming,3=Other,4=Social Media): "))

family_comm = int(input("Family Communication Level: "))
weekend_usage = float(input("Weekend Usage Hours: "))

features = np.array([[
    age,
    gender,
    daily_usage,
    sleep_hours,
    social_interactions,
    exercise_hours,
    anxiety,
    depression,
    self_esteem,
    parental_control,
    screen_before_bed,
    phone_checks,
    apps_used,
    time_social,
    time_gaming,
    time_education,
    purpose,
    family_comm,
    weekend_usage
]])

features_scaled = scaler.transform(features)

start_time = time.time()

prediction = model.predict(features_scaled)
probabilities = model.predict_proba(features_scaled)

end_time = time.time()
class_map = {
    0: "High",
    1: "Low",
    2: "Medium"
}

predicted_class = class_map[prediction[0]]

confidence = np.max(probabilities) * 100

print("\n===========================")
print("Prediction Result")
print("===========================")

print("Predicted Addiction Level:", predicted_class)

print("Confidence: {:.2f}%".format(confidence))

print("Time Taken: {:.6f} seconds".format(end_time - start_time))