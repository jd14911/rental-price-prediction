import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
startTime = datetime.now()

# Wczytanie modelu i scalera
model_filename = "rental_price_model.sv"
scaler_filename = "scaler.pkl"
model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Wczytanie danych z pliku CSV
data = pd.read_csv("apartments_rent_pl_2024_04.csv")

# Wyznaczenie minimalnych i maksymalnych wartości dla zmiennych
powierzchnia_min, powierzchnia_max = data["squareMeters"].min(), data["squareMeters"].max()
liczba_pokoi_min, liczba_pokoi_max = data["rooms"].min(), data["rooms"].max()
pietro_min, pietro_max = data["floor"].min(), data["floor"].max()
liczba_pieter_min, liczba_pieter_max = data["floorCount"].min(), data["floorCount"].max()
odleglosc_min, odleglosc_max = data["centreDistance"].min(), data["centreDistance"].max()

# Kodowanie zmiennych
city_d = {0: 'bialystok', 1: 'bydgoszcz', 2: 'czestochowa', 3: 'gdansk', 4: 'gdynia', 5: 'katowice', 6: 'krakow', 7: 'lodz', 8: 'lublin', 9: 'poznan', 10: 'radom', 11: 'rzeszow', 12: 'szczecin', 13: 'warszawa', 14: 'wroclaw'}
boolean_d = {1: "Tak", 0: "Nie"}

# Mapowanie miast na zdjęcia
city_images = {
    'bialystok': "https://upload.wikimedia.org/wikipedia/commons/7/7a/Rynek_Ko%C5%9Bciuszki%2C_Bia%C5%82ystok_%282%29.jpg",
    'bydgoszcz': "https://upload.wikimedia.org/wikipedia/commons/a/ae/Panorama_SM_Bydgoszcz_soft.jpg",
    'czestochowa': "https://upload.wikimedia.org/wikipedia/commons/e/ea/Cz%C4%99stochowa_%28B%C5%82eszno%29_bloki.jpg",
    'gdansk': "https://upload.wikimedia.org/wikipedia/commons/5/5e/Gda%C5%84sk_20240625_200527.jpg",
    'gdynia': "https://upload.wikimedia.org/wikipedia/commons/c/c9/Hotel_Mercure_w_Gdyni_-_grudzie%C5%84_2023.jpg",
    'katowice': "https://upload.wikimedia.org/wikipedia/commons/5/5b/Katowice_city_centre_%286%29.jpg",
    'krakow': "https://upload.wikimedia.org/wikipedia/commons/9/92/Cracovia_%28Polonia%29_Krak%C3%B3w_%28Polska%29._-_52631980745.jpg",
    'lodz': "https://upload.wikimedia.org/wikipedia/commons/5/54/20190730_161646_View_of_Lodz_30-07-2019_Piotrkowska_street.jpg",
    'lublin': "https://upload.wikimedia.org/wikipedia/commons/1/15/Lublin_2024%2C_widoki_z_Don%C5%BCona_%E2%80%93_Stare_Miasto.jpg",
    'poznan': "https://upload.wikimedia.org/wikipedia/commons/c/c4/Pozna%C5%84_Centrum.jpg",
    'radom': "https://upload.wikimedia.org/wikipedia/commons/0/04/Dom_towarowy_Senior_Radom_6.jpg",
    'rzeszow': "https://upload.wikimedia.org/wikipedia/commons/9/94/Hala_Podpromie.JPG",
    'szczecin': "https://upload.wikimedia.org/wikipedia/commons/a/a6/PolandSzczecinPanorama.JPG",
    'warszawa': "https://upload.wikimedia.org/wikipedia/commons/9/96/Panorama_ul._Emilii_Plater_w_Warszawie_radek_ko%C5%82akowski.jpg",
    'wroclaw': "https://upload.wikimedia.org/wikipedia/commons/3/37/Wroclaw-cathedral-island-107.JPG"
}

# Funkcja główna aplikacji
def main():

    st.set_page_config(page_title="Predykcja wynajmu mieszkań")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    with overview:
        st.title("Prognoza ceny wynajmu mieszkań")
        st.write("Aplikacja umożliwia oszacowanie ceny wynajmu mieszkań na podstawie podanych parametrów.")

    with left:
        miasto = st.selectbox("Miasto", list(city_d.values()))
        powierzchnia = st.number_input("Powierzchnia (m²)", min_value=int(powierzchnia_min), max_value=int(powierzchnia_max))
        liczba_pokoi = st.slider("Liczba pokoi", min_value=int(liczba_pokoi_min), max_value=int(liczba_pokoi_max))
        odleglosc = st.number_input("Odległość od centrum (km)", min_value=float(odleglosc_min), max_value=float(odleglosc_max))

        # Wyświetlenie zdjęcia miasta
        if miasto in city_images:
            st.image(city_images[miasto], caption=f"Widok miasta: {miasto.capitalize()}")

    with right:
        pietro = st.slider("Które piętro", min_value=int(pietro_min), max_value=int(pietro_max))
        liczba_pieter = st.slider("Ilość pięter budynku", min_value=int(liczba_pieter_min), max_value=int(liczba_pieter_max))
        balkon = st.radio("Czy jest balkon?", list(boolean_d.values()))
        parking = st.radio("Czy ma parking?", list(boolean_d.values()))
        winda = st.radio("Czy jest winda?", list(boolean_d.values()))
        ochrona = st.radio("Czy jest ochrona?", list(boolean_d.values()))

    # Konwersja cech do kodowania numerycznego
    miasto_idx = list(city_d.values()).index(miasto)
    balkon_flag = 1 if balkon == "Tak" else 0
    parking_flag = 1 if parking == "Tak" else 0
    winda_flag = 1 if winda == "Tak" else 0
    ochrona_flag = 1 if ochrona == "Tak" else 0

    # Przygotowanie danych wejściowych
    input_data = np.array([[miasto_idx, powierzchnia, liczba_pokoi, odleglosc, pietro, liczba_pieter, balkon_flag, parking_flag, winda_flag, ochrona_flag]], dtype=np.float64)

    # Normalizacja danych
    input_normalized = scaler.transform(input_data)

    # Predykcja
    if st.button("Oszacuj cenę"):
        prediction = model.predict(input_normalized)
        st.subheader(f"Przewidywana cena wynajmu: {float(prediction[0]):.2f} PLN")
      
    # Informacje o zbiorze danych
    st.subheader("Informacje o zbiorze danych")
    st.write("Źródło danych: Kaggle (https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland).")
    st.write("Zbiór danych zawiera informacje o cenach wynajmu mieszkań w Polsce z kwietnia 2024 roku. Liczba rekordów: 9,484")

    # Informacje o modelu
    st.subheader("Informacje o modelu")
    st.write("Model to sieć neuronowa z warstwami Dense (Keras/TensorFlow). Model został wytrenowany oraz znormalizowany przy pomocy `MinMaxScaler`.")
    st.write("Główne metryki modelu to:")
    st.write("- Średni błąd bezwzględny (MAE): 820.3614501953125 PLN")
    st.write("- Zakres cen w zbiorze: od  412 do 18 700 PLN.")

if __name__ == "__main__":
    main()