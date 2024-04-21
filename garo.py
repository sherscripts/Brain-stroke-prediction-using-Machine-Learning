from flask import Flask, render_template, request, jsonify
import pickle
from pathlib import Path
from flask_cors import CORS

file_path = Path(__file__).resolve().parent / 'GARO.pkl'
model = pickle.load(open(file_path, 'rb'))

feature_importances = model.feature_importances_
feature_names = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
       'gender_Female', 'gender_Male', 'gender_Other', 'ever_married_No',
       'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
       'work_type_Private', 'work_type_Self-employed', 'work_type_children',
       'Residence_type_Rural', 'Residence_type_Urban',
       'smoking_status_Unknown', 'smoking_status_formerly smoked',
       'smoking_status_never smoked', 'smoking_status_smokes']
min_values = [0.08, 0, 0, 55.22, 10.3, 0, 0, 0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
max_values = [92.00, 1, 1, 293,49.9, 1, 1, 1, 1, 1, 1,1,1, 1, 1, 1, 1, 1, 1, 1, 1]

garo = Flask(__name__)
CORS(garo)

@garo.route('/feature_names', methods=['GET']) 
def get_feature_names():
    return jsonify({"feature_names": feature_names})

@garo.route('/predict', methods=['POST'])
def predict():
    print(request.headers)
    if request.method == 'POST':
        if request.headers['Content-Type'] == 'application/json':
            try:
                data = request.json
                age = int(data['age'])
                hypertension = int(data['hypertension'])
                heart_disease = int(data['heart_disease'])
                avg_glucose_level = float(data['avg_glucose_level'])
                bmi = float(data['bmi'])
                gender = data['gender']
                married = data['married']
                work = data['work']
                residence = data['residence']
                smoking = data['smoking']

                # Convert categorical variables to binary
                gender_Male = 1 if gender == "Male" else 0
                gender_Female = 1 if gender == "Female" else 0
                gender_Other = 1 if gender == "Other" else 0

                ever_married_Yes = 1 if married == "Yes" else 0
                ever_married_No = 1 if married == "No" else 0

                

                if work == 'Self-employed':
                    work_type_Govt_job = 0
                    work_type_Private = 0
                    work_type_Self_employed = 1
                    work_type_children = 0
                    work_type_Never_worked=0
                elif work == 'Private':
                    work_type_Govt_job = 0
                    work_type_Private = 1
                    work_type_Self_employed = 0
                    work_type_children = 0
                    work_type_Never_worked=0
                elif work == "children":
                    work_type_Govt_job = 0
                    work_type_Private = 0
                    work_type_Self_employed = 0
                    work_type_children = 1
                    work_type_Never_worked=0
                elif work == "Never_worked":
                    work_type_Govt_job = 0
                    work_type_Private = 0
                    work_type_Self_employed = 0
                    work_type_children = 0
                    work_type_Never_worked=1
                else:
                    work_type_Govt_job = 1
                    work_type_Private = 0
                    work_type_Self_employed = 0
                    work_type_children = 0
                    work_type_Never_worked=0

                Residence_type_Urban = 1 if residence == "Urban" else 0
                Residence_type_Rural = 1 if residence == "Rural" else 0

                if smoking == 'formerly smoked':
                    smoking_status_Unknown = 0
                    smoking_status_formerly_smoked = 1
                    smoking_status_never_smoked = 0
                    smoking_status_smokes = 0
                elif smoking == 'smokes':
                    smoking_status_Unknown = 0
                    smoking_status_formerly_smoked = 0
                    smoking_status_never_smoked = 0
                    smoking_status_smokes = 1
                elif smoking == "never smoked":
                    smoking_status_Unknown = 0
                    smoking_status_formerly_smoked = 0
                    smoking_status_never_smoked = 1
                    smoking_status_smokes = 0
                else:
                    smoking_status_Unknown = 1
                    smoking_status_formerly_smoked = 0
                    smoking_status_never_smoked = 0
                    smoking_status_smokes = 0

                print("Values after processing:")
                print(f"age: {age}")
                print(f"hypertension: {hypertension}")
                print(f"heart_disease: {heart_disease}")
                print(f"avg_glucose_level: {avg_glucose_level}")
                print(f"bmi: {bmi}")
                print(f"gender_Female: {gender_Female}")
                print(f"gender_Male: {gender_Male}")
                print(f"ever_married_No: {ever_married_No}")
                print(f"ever_married_Yes: {ever_married_Yes}")
                print(f"work_type_Govt_job: {work_type_Govt_job}")
                print(f"work_type_Private: {work_type_Private}")
                print(f"work_type_Self_employed: {work_type_Self_employed}")
                print(f"work_type_children: {work_type_children}")
                print(f"Residence_type_Rural: {Residence_type_Rural}")
                print(f"Residence_type_Urban: {Residence_type_Urban}")
                print(f"smoking_status_Unknown: {smoking_status_Unknown}")
                print(f"smoking_status_formerly_smoked: {smoking_status_formerly_smoked}")
                print(f"smoking_status_never_smoked: {smoking_status_never_smoked}")
                print(f"smoking_status_smokes: {smoking_status_smokes}")

                # Make prediction using the model
                result = model.predict([[age, hypertension, heart_disease, avg_glucose_level, bmi, gender_Female, gender_Male,gender_Other,ever_married_No, ever_married_Yes, work_type_Govt_job,work_type_Never_worked,work_type_Private, work_type_Self_employed,
                                        work_type_children, Residence_type_Rural, Residence_type_Urban, smoking_status_Unknown,
                                        smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes]])[0]

                # Calculate feature contributions based on user input
                user_input_values = [age, hypertension, heart_disease, avg_glucose_level, bmi, gender_Female, gender_Male,gender_Other,
                                        ever_married_No, ever_married_Yes, work_type_Govt_job,work_type_Never_worked, work_type_Private, work_type_Self_employed,
                                        work_type_children, Residence_type_Rural, Residence_type_Urban, smoking_status_Unknown,smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes]

                feature_contributions = {feature: (user_input_values[i] - min_val) / (max_val - min_val) * importance
    for i, (feature, min_val, max_val, importance) in enumerate(
        zip(feature_names, min_values, max_values, feature_importances)
    )
                }
    # Normalize contributions
                total_contribution = sum(feature_contributions.values())
                normalized_contributions = {feature: (contribution / total_contribution) * 100 for feature, contribution in feature_contributions.items()
}

                result_message = "No stroke" if result == 0 else "Stroke"
                return jsonify({"result_message": result_message, "contributions": normalized_contributions})


                

            except Exception as e:
                print("Error parsing JSON:", e)
                return jsonify({"error": "Invalid JSON data"}), 400
        else:
            # If the content type is not JSON, handle it accordingly
            return jsonify({"error": "Unsupported Media Type"}), 415 
    else:
        # Handle GET requests if needed
        return render_template('index.html', **locals())

if __name__ == '__main__':
    garo.run(port=5000, debug=True)