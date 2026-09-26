from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Prediction
from django.db.models import Count

import numpy as np
from .ml_model import model, scaler, class_map
import random

class0 = [
    "Kapalbati pranayama - detoxifies mind",
    "Bhramari pranayama - reduces anxiety",
    "Yoga nidra - deep relaxation"
]

class1 = [
    "Tadasana - improves focus",
    "Vrikshasana - increases concentration",
    "Bhujangasana - reduces stress"
]

class2 = [
    "Surya Namaskar - improves balance",
    "Anulom Vilom - calms mind",
    "Meditation - mindfulness"
]

@login_required
def dashboard_view(request):
    return render(request, 'dashboard/dashboard.html')

@login_required
def predict_view(request):

    result = None
    confidence = None
    suggestion = None

    if request.method == "POST":

        age = int(request.POST.get("age"))
        gender = int(request.POST.get("gender"))
        daily_usage = float(request.POST.get("daily_usage"))
        sleep_hours = float(request.POST.get("sleep_hours"))
        social_interactions = int(request.POST.get("social_interactions"))
        exercise_hours = float(request.POST.get("exercise_hours"))
        anxiety = int(request.POST.get("anxiety"))
        depression = int(request.POST.get("depression"))
        self_esteem = int(request.POST.get("self_esteem"))
        parental_control = int(request.POST.get("parental_control"))
        screen_before_bed = float(request.POST.get("screen_before_bed"))
        phone_checks = int(request.POST.get("phone_checks"))
        apps_used = int(request.POST.get("apps_used"))
        time_social = float(request.POST.get("time_social"))
        time_gaming = float(request.POST.get("time_gaming"))
        time_education = float(request.POST.get("time_education"))
        purpose = int(request.POST.get("purpose"))
        family_comm = int(request.POST.get("family_comm"))
        weekend_usage = float(request.POST.get("weekend_usage") or 0)

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

        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        predicted_class=int(prediction[0])
        class_map={
            0: "High",
            1: "Low",
            2: "Medium"
        }

        result = class_map[prediction[0]]
        confidence = float(np.max(probabilities) * 100)
        predicted_class = prediction[0]
        if predicted_class == 0:
            suggestion = random.choice(class0)
        elif predicted_class == 1:
            suggestion = random.choice(class1)
        elif predicted_class == 2:
            suggestion = random.choice(class2)
        

        # Save input data
        input_data = {
            "age": age,
            "gender": gender,
            "daily_usage": daily_usage,
            "sleep_hours": sleep_hours,
            "social_interactions": social_interactions,
            "exercise_hours": exercise_hours,
            "anxiety": anxiety,
            "depression": depression,
            "self_esteem": self_esteem,
            "parental_control": parental_control,
            "screen_before_bed": screen_before_bed,
            "phone_checks": phone_checks,
            "apps_used": apps_used,
            "time_social": time_social,
            "time_gaming": time_gaming,
            "time_education": time_education,
            "purpose": purpose,
            "family_comm": family_comm,
            "weekend_usage": weekend_usage
        }

        # Save to database
        Prediction.objects.create(
            user=request.user,
            input_data=input_data,
            predicted_class=result,
            confidence=confidence
        )

    return render(request, "dashboard/predict.html", {
        "result": result,
        "confidence": confidence,
        "suggestion": suggestion
    })

@login_required
def history_view(request):
    qs = Prediction.objects.filter(user=request.user)

    # Count per class
    class_counts = qs.values('predicted_class').annotate(count=Count('id'))

    labels = [item['predicted_class'] for item in class_counts]
    data = [item['count'] for item in class_counts]

    return render(request, 'dashboard/history.html', {'labels': labels,'data': data,})

@login_required
def profile_page(request):
    profile = request.user.profile
    return render(request, 'dashboard/profile.html', {'profile': profile})

@login_required
def my_predictions(request):
    predictions = Prediction.objects.filter(user=request.user)
    return render(request, 'dashboard/my_predictions.html', {'predictions': predictions})
