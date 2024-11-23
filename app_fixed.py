# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
sex_d = {0: "Kobieta", 1: "Mężczyzna"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

data = pd.read_csv("DSP_1.csv")
# Wczytujemy dane z pliku CSV

age_min, age_max = data["age"].min(), data["age"].max()
sibsp_min, sibsp_max = data["sibsp"].min(), data["sibsp"].max()
parch_min, parch_max = data["parch"].min(), data["parch"].max()
fare_min, fare_max = data["fare"].min(), data["fare"].max()
# Wyznaczamy wartości minimalne i maksymalne dla zmiennych

def main():

	st.set_page_config(page_title="Titanic")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg")

	with overview:
		st.title("Titanic")

	with left:
		pclass_radio = st.radio( "Klasa", list(pclass_d.keys()), format_func=lambda x: pclass_d[x] )
		sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
		embarked_radio = st.radio( "Port zaokrętowania", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )

	with right:
		age_slider = st.slider("Wiek", value=age_min, min_value=age_min, max_value=age_max)
		sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", min_value=sibsp_min, max_value=sibsp_max)
		parch_slider = st.slider("Liczba rodziców i/lub dzieci", min_value=parch_min, max_value=parch_max)
		fare_slider = st.slider("Cena biletu", min_value=fare_min, max_value=fare_max)

	data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba przeżyłaby katastrofę?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
