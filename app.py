import streamlit as st
import joblib
import pandas as pd

import os


st.set_page_config(page_title="Dashboard Proiect ML", layout="wide")

@st.cache_resource
def load_models_regresie():
    return {
        "XGBoost": joblib.load('notebooks/regresie_files/modele/model_xgb.pkl'),
        "CatBoost": joblib.load('notebooks/regresie_files/modele/model_cat.pkl'),
        "Random Forest": joblib.load('notebooks/regresie_files/modele/model_rf.pkl'),
        "EBM": joblib.load('notebooks/regresie_files/modele/model_ebm.pkl'),
        "Decision Tree": joblib.load('notebooks/regresie_files/modele/model_dt.pkl')
    }

@st.cache_resource
def load_models_clasificare():
    return {
        "models": {
            "CatBoost": joblib.load('notebooks/clasificare_files/modele/model_cat.pkl'),
            "EBM": joblib.load('notebooks/clasificare_files/modele/model_ebm.pkl'),
            "Logistic Regression": joblib.load('notebooks/clasificare_files/modele/model_lr.pkl'),
            "XGBoost": joblib.load('notebooks/clasificare_files/modele/model_xgb.pkl'),
            "Naive Bayes": joblib.load('notebooks/clasificare_files/modele/model_nb.pkl')
        },
        "scaler": joblib.load('notebooks/scaler_stroke.pkl')
        }

st.sidebar.title("Navigare")
pagina = st.sidebar.radio("Mergi la:", ["Clasificare(Risc atac cerebral)", "Regresie (Preturi masini)"])

if pagina == "Regresie (Preturi masini)":
    st.title("Predictie preturi masini second-hand")

    st.info("""
    **Context:** Acest modul estimeaza pretul de vanzare al masinilor bazat pe diferite caracteristici
    **Dataset:** Setul de date contine informatii despre putere, an, kilometraj, combustibil etc.
    """)

    with st.expander("Vezi analiza exploratorie(EDA)"):
        st.subheader("Distributia datelor si corelatii")
        col_eda1, col_eda2 = st.columns(2)

        try:
            col_eda1.image("notebooks/regresie_files/eda_outliers.png", caption="Detectie outlieri (Preț și KM)")
            col_eda2.image("notebooks/regresie_files/eda_correlation.png", caption="Matricea de corelatie")
        except:
            st.warning("Imaginile EDA nu au fost gasite în folderul 'notebooks'.")




    models=load_models_regresie()
    selected_model_name = st.selectbox("Alege modelul de testat:", list(models.keys()))
    model = models[selected_model_name]

    with st.expander("Vezi detaliile tehnice ale modelului (Hiperparametri)"):
        st.json(model.get_params())

    st.subheader("Introdu datele pentru predictie:")
    c1, c2, c3 = st.columns(3)
    with c1:
        year = st.number_input("An fabricatie", 1990, 2024, 2018)
        max_power = st.number_input("Cai putere", 30, 600, 110)
        km_driven = st.number_input("Kilometraj (km)", 0, 1000000, 50000)
    with c2:
        engine = st.number_input("Capacitate motor (cc)", 600, 6000, 1500)
        seats = st.selectbox("Numar locuri", [2, 4, 5, 7, 8], index=2)
        mileage = st.number_input("Consum (km/l)", 5.0, 40.0,18.0)
    with c3:
        fuel = st.selectbox("Combustibil", ["Motorina", "Benzina", "GPL"])
        transmission = st.selectbox("Transmisie", ["Manuala", "Automata"])
        seller_type = st.selectbox("Tip vanzator", ["Individual", "Reprezentanta"])
        owner = st.selectbox("Proprietar",
                             ["Prim proprietar", "Al doilea proprietar", "Al treilea proprietar", "Al patrulea sau mai mult"])

    if st.button("Predict"):
        expected_columns = [
            'year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
            'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol',
            'seller_type_Individual', 'seller_type_Trustmark Dealer',
            'transmission_Manual', 'owner_Fourth & Above Owner',
            'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner'
        ]
        input_data = {col: 0 for col in expected_columns}

        input_data['year'] = year
        input_data['km_driven'] = km_driven
        input_data['mileage'] = mileage
        input_data['engine'] = engine
        input_data['max_power'] = max_power
        input_data['seats'] = seats


        if fuel == "Motorina":
            input_data['fuel_Diesel'] = 1
        elif fuel == "GPL":
            input_data['fuel_LPG'] = 1
        elif fuel == "Benzina":
            input_data['fuel_Petrol'] = 1

        if seller_type == "Individual":
            input_data['seller_type_Individual'] = 1
        elif seller_type == "Reprezentanta":
            input_data['seller_type_Trustmark Dealer'] = 1

        if transmission == "Manuala":
            input_data['transmission_Manual'] = 1

        if owner == "Al doilea proprietar":
            input_data['owner_Second Owner'] = 1
        elif owner == "Al treilea proprietar":
            input_data['owner_Third Owner'] = 1
        elif owner == "Al patrulea sau mai mult":
            input_data['owner_Fourth & Above Owner'] = 1

        features_df = pd.DataFrame([input_data])[expected_columns]
        prediction = model.predict(features_df)[0]
        st.success(f"Pret estimat: {prediction:,.2f} unitati")

    st.divider()
    col_lc, col_shap = st.columns(2)
    model_name_lower = selected_model_name.lower().replace(" ", "_")

    with col_lc:
        st.subheader("Curbe de invatare")
        lc_path = f"notebooks/regresie_files/lc_{model_name_lower}.png"
        if os.path.exists(lc_path):
            st.image(lc_path, caption=f"Performanta modelului {selected_model_name}")
        else:
            st.info(f"Graficul curbei de invatare pentru {selected_model_name} va fi generat curand.")

    with col_shap:
        st.subheader("Explicabilitate SHAP")
        shap_path = f"notebooks/regresie_files/shap_{model_name_lower}.png"

        if os.path.exists(shap_path):
            st.image(shap_path, caption=f"Importanta caracteristicilor conform SHAP ({selected_model_name})")
        else:
            st.info("Analiza SHAP globala este disponibila pentru primele 3 modele (XGB, Cat, RF).")
else:
    st.title("Clasificare: Detectie risc atac cerebral")
    st.info ("""
    **Context:** Aceasta sectiune clasifica pacientii in functie de riscul de a suferi un accident vascular cerebral.
    **Dataset:** Setul de date contine variabile clinice(hipertensiune, nivelul de glucoza din sange, BMI) si demografice (varsta etc.).
    **Obiectiv** Minimizarea cazurilor de fals negativ pentru a nu omite pacientii aflati in zona de risc.""")

    with st.expander("Vezi analiza exploratorie (EDA)"):
        st.subheader("Distributia factorilor de risc")
        col_eda1, col_eda2 = st.columns(2)


        col_eda1.image("notebooks/clasificare_files/eda_correlation_clasificare.png", caption="Matricea de corelatie")
        col_eda2.image("notebooks/clasificare_files/eda_outliers_clasificare.png", caption="Detectie outlieri")


    assets = load_models_clasificare()
    model_cls =assets["models"]
    scaller=assets["scaler"]
    selected_model_name = st.selectbox("Alege modelul de testat:", list(model_cls.keys()))
    model = model_cls[selected_model_name]

    metrics_map = {
        "CatBoost": {"Recall": "81%", "ROC-AUC": "0.86", "F1": "0.78", "Acc": "85%"},
        "EBM": {"Recall": "79%", "ROC-AUC": "0.84", "F1": "0.76", "Acc": "83%"},
        "Logistic Regression": {"Recall": "75%", "ROC-AUC": "0.82", "F1": "0.72", "Acc": "81%"},
        "XGBoost": {"Recall": "77%", "ROC-AUC": "0.83", "F1": "0.74", "Acc": "82%"},
        "Naive Bayes": {"Recall": "88%", "ROC-AUC": "0.79", "F1": "0.68", "Acc": "70%"}
    }

    with st.expander("Vezi detaliile tehnice (Hiperparametrii)"):
        st.json(model.get_params())
        st.write(f"Performanta model ({selected_model_name})")
        m = metrics_map.get(selected_model_name, {"Recall": "-", "ROC-AUC": "-", "F1": "-", "Acc": "-"})
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Recall", m["Recall"])
        col_m2.metric("ROC-AUC", m["ROC-AUC"])
        col_m3.metric("F1-Score", m["F1"])
        col_m4.metric("Acuratete", m["Acc"])


    st.subheader("Introdu datele pacientului pentru analiza:")
    c1,c2,c3=st.columns(3)

    with c1:
        age = st.slider("Varsta pacientului", 0, 100, 50)
        gender = st.selectbox("Gen", ["Masculin", "Feminin"])
        avg_glucose = st.number_input("Nivel mediu glucoza (mg/dL)", 50.0, 300.0, 90.0)
        bmi = st.number_input("Indice masa corporala (BMI)", 10.0, 60.0, 28.0)
    with c2:
        hypertension = st.selectbox("Hipertensiune", ["Nu", "Da"])
        heart_disease = st.selectbox("Boli de inima", ["Nu", "Da"])
        residence = st.selectbox("Mediu rezidenta", ["Urban", "Rural"])
    with c3:
        work_type = st.selectbox("Tip munca", ["Privat", "Self-employed", "Stat", "Copil", "Niciodata"])
        smoking = st.selectbox("Statut fumator", ["Fost fumator", "Niciodata", "Fumator", "Necunoscut"])
        married = st.selectbox("Casatorit(a)?", ["Da", "Nu"])

    if st.button("Calculeaza risc"):
        mapping = {"Nu": 0, "Da": 1, "Rural": 0, "Urban": 1}

        expected_columns = [
            'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
            'gender_Male', 'ever_married_Yes',
            'work_type_Never_worked',
            'work_type_Private', 'work_type_Self-employed', 'work_type_children',
            'Residence_type_Urban',
            'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
        ]
        input_dict = {col: 0 for col in expected_columns}
        input_dict['age'] = age
        input_dict['hypertension'] = mapping[hypertension]
        input_dict['heart_disease'] = mapping[heart_disease]
        input_dict['avg_glucose_level'] = avg_glucose
        input_dict['bmi'] = bmi
        if gender == "Masculin": input_dict['gender_Male'] = 1
        if married == "Da": input_dict['ever_married_Yes'] = 1
        input_dict['Residence_type_Urban'] = mapping[residence]

        if work_type == "Privat":
            input_dict['work_type_Private'] = 1
        elif work_type == "Self-employed":
            input_dict['work_type_Self-employed'] = 1
        elif work_type == "Copil":
            input_dict['work_type_children'] = 1

        if smoking == "Fost fumator":
            input_dict['smoking_status_formerly smoked'] = 1
        elif smoking == "Niciodata":
            input_dict['smoking_status_never smoked'] = 1
        elif smoking == "Fumator":
            input_dict['smoking_status_smokes'] = 1

        features_df = pd.DataFrame([input_dict])[expected_columns]
        numeric_features = ['age', 'avg_glucose_level', 'bmi']
        features_df[numeric_features] = scaller.transform(features_df[numeric_features])

        prediction = model.predict(features_df)[0]
        prob = model.predict_proba(features_df)[0][1] if hasattr(model, "predict_proba") else None

        if prediction == 1:
            st.error(f"Atentie: risc ridicat de AVC!(Probabilitate: {prob:.2%})")
        else:
            st.success(f"Risc scazut detectat. (Probabilitate: {prob:.2%})")

    st.divider()
    col_lc, col_shap = st.columns(2)
    model_keys = {"CatBoost": "catboost", "EBM": "ebm", "Logistic Regression": "logreg", "XGBoost": "xgboost",
                  "Naive Bayes": "naive_bayes"}
    sufix = model_keys[selected_model_name]

    with col_lc:
        st.subheader("Curbe de invatare")
        lc_name = "learning_curve_log._regression.png" if sufix == "logreg" else f"learning_curve_{sufix}.png"
        lc_path = f"notebooks/clasificare_files/{lc_name}"
        if os.path.exists(lc_path):
            st.image(lc_path, caption=f"Performanta {selected_model_name}")
        else:
            st.warning(f"Fisierul {lc_path} lipseste.")

    with col_shap:
        st.subheader("Explicabilitate SHAP")
        if sufix in ["catboost", "ebm", "logreg"]:
            st.image(f"notebooks/clasificare_files/shap_summary_{sufix}.png", caption="Impactul global al variabilelor")
            st.image(f"notebooks/clasificare_files/shap_bar_{sufix}.png",
                     caption="Importanta medie a caracteristicilor")

            st.markdown("**De ce a decis modelul asta? (Exemplu Local)**")
            st.image(f"notebooks/clasificare_files/shap_waterfall_{sufix}.png",
                     caption="Waterfall Plot: Contributia fiecarei valori")
            st.image(f"notebooks/clasificare_files/shap_force_{sufix}.png",
                     caption="Force Plot")

            st.markdown("**Analiza de dependenta (Varsta & Glucoza)**")
            cs1, cs2 = st.columns(2)
            cs1.image(f"notebooks/clasificare_files/shap_scatter_age_{sufix}.png")
            cs2.image(f"notebooks/clasificare_files/shap_scatter_avg_glucose_level_{sufix}.png")

        else:
            st.info("SHAP disponibil doar pentru primele 3 modele (CatBoost, EBM, LogReg).")



