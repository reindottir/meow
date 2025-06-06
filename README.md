# Проект: Предиктивное обслуживание оборудования
## Цель проекта
Разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования (Target = 1) или нет (Target = 0).
Результаты оформлены в виде интерактивного Streamlit-приложения.
## Описание
Streamlit-приложение для прогнозирования отказов оборудования с использованием:
- Random Forest
- Streamlit для интерфейса
- UCI Predictive Maintenance Dataset

## Установка
```bash
git clone https://github.com/PWU3U/predictive_maintenance_project.git
cd predictive_maintenance_project
pip install -r requirements.txt
streamlit run app.py
