import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    st.title("Анализ данных и модель")
    
    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите данные (CSV)", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        st.warning("Используются демонстрационные данные")
        data = pd.read_csv("data/predictive_maintenance.csv")
    
    # Предобработка
    data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    data['Type'] = LabelEncoder().fit_transform(data['Type'])
    
    # Разделение данных
    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Масштабирование
    scaler = StandardScaler()
    num_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
               'Torque [Nm]', 'Tool wear [min]']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Оценка
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Визуализация
    st.header("Результаты")
    st.write(f"Accuracy: {accuracy:.4f}")
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    
    # Интерфейс предсказания
    st.header("Предсказание")
    with st.form("input_form"):
        type_val = st.selectbox("Тип оборудования", ['L', 'M', 'H'])
        air_temp = st.number_input("Температура воздуха [K]", value=300.0)
        process_temp = st.number_input("Температура процесса [K]", value=310.0)
        rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500)
        torque = st.number_input("Крутящий момент [Nm]", value=40.0)
        tool_wear = st.number_input("Износ инструмента [min]", value=100)
        
        if st.form_submit_button("Предсказать"):
            input_data = pd.DataFrame({
                'Type': [type_val],
                'Air temperature [K]': [air_temp],
                'Process temperature [K]': [process_temp],
                'Rotational speed [rpm]': [rotational_speed],
                'Torque [Nm]': [torque],
                'Tool wear [min]': [tool_wear]
            })
            input_data['Type'] = LabelEncoder().fit_transform(input_data['Type'])
            input_data[num_cols] = scaler.transform(input_data[num_cols])
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1]
            st.success(f"Отказ оборудования: {'Да' if prediction == 1 else 'Нет'}")
            st.info(f"Вероятность отказа: {proba:.4f}")

if __name__ == "__main__":
    main()