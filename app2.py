# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

#import pathlib
#from pathlib import Path

#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath
#Konieczne było usunięcie path, ponieważ generowało błędy w streamlit share

filename = "model2.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

restingecg_d = {0:"Normalne",1:"ST", 2:"LVH"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

data = pd.read_csv("DSP_8.csv")
# Wczytujemy dane z pliku CSV

age_min, age_max = data["Age"].min(), data["Age"].max()
restingbp_min, restingbp_max = data["RestingBP"].min(), data["RestingBP"].max()
cholesterol_min, cholersterol_max = data["Cholesterol"].min(), data["Cholesterol"].max()
fastingbs_min, fastingbs_max = data["FastingBS"].min(), data["FastingBS"].max()
maxhr_min, maxhr_max = data["MaxHR"].min(), data["MaxHR"].max()
oldpeak_min, oldpeak_max = data["Oldpeak"].min(), data["Oldpeak"].max()
# Wyznaczamy wartości minimalne i maksymalne dla zmiennych

def main():

	st.set_page_config(page_title="Titanic")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://upload.wikimedia.org/wikipedia/commons/d/d7/Simple-heart-icon.png")

	with overview:
		st.title("Choroby Serca")

	with left:
		pclass_radio = st.radio( "ECG Spoczynkowe", list(restingecg_d.keys()), format_func=lambda x: restingecg_d[x] )
		age_slider = st.slider("Wiek", value=age_min, min_value=age_min, max_value=age_max)
		restingbp_slider = st.slider("Spoczynkowe BP", min_value=restingbp_min, max_value=restingbp_max)
		cholesterol_slider = st.slider("Cholesterol", min_value=cholesterol_min, max_value=cholersterol_max)
	with right:		
		fastingbs_slider = st.slider("Poziom cukru we krwi na czczo", min_value=fastingbs_min, max_value=fastingbs_max)
		maxhr_slider = st.slider("Maksymalne tętno", min_value=maxhr_min, max_value=maxhr_max)
		oldpeak_slider = st.slider("Oldpeak", min_value=oldpeak_min, max_value=oldpeak_max)

	data = [[pclass_radio, age_slider, restingbp_slider, cholesterol_slider, fastingbs_slider, maxhr_slider, oldpeak_slider]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba posiada chorobę serca?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
