import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from streamlit_folium import folium_static

import io
import pickle

#from gsheetsdb import connect

from methodes_angers_bus_tram import *

#conn = connect()

#def run_query(query):
#    rows = conn.execute(query, headers=1)
#    rows = rows.fetchall()
#    return rows

#@st.cache(suppress_st_warning=True, allow_output_mutation=True) 
#def load_data():
#    df_global = pd.read_csv("C:/Users/clovi/Documents/GitHub/qualite_trafic_tram_bus_angers/angers-bus-tram_join.csv", encoding="utf-8")
#    df_propre = pd.read_csv("C:/Users/clovi/Documents/GitHub/qualite_trafic_tram_bus_angers/bus_trafic_clean.csv", encoding="utf-8")
#    df_propre = df_propre.astype({"horodatage": "datetime64", 
#                "Heure_estimee_de_passage_a_L_arret": "datetime64", 
#                "date_heure" : "datetime64", 
#                "date": "datetime64",
#                "date_heure" : "datetime64",
#                })
#    df_meteo = pd.read_csv("C:/Users/clovi/Documents/GitHub/qualite_trafic_tram_bus_angers/export-angers_meteo.csv", sep=";", encoding="utf-8")
#    
#    df_arret = pd.read_csv("C:/Users/clovi/Documents/GitHub/qualite_trafic_tram_bus_angers/stops.txt", sep=",")
#    
#    df_result_ML = pd.read_csv("C:/Users/clovi/Documents/GitHub/qualite_trafic_tram_bus_angers/result.csv")
#    
#    df_fichier_pour_pred_ML1 = pd.read_excel("C:/Users/clovi/Documents/GitHub/qualite_trafic_tram_bus_angers/fichier_pour_pred_ML1.xlsx")
#    return df_global, df_propre, df_meteo, df_arret, df_result_ML, df_fichier_pour_pred_ML1

@st.cache(suppress_st_warning=True)
def load_data():
    df_global_url = "https://drive.google.com/file/d/1DMJGzA92Mwyp0-nS5LSMWZ4lCoMLzLQ4/view?usp=sharing"
    file_id=df_global_url.split('/')[-2]
    dwn_url='https://drive.google.com/uc?export=download&confirm=s5vl&id=' + file_id
    #st.write(dwn_url)
    df_global = pd.read_csv(dwn_url)

    df_propre_path = "https://drive.google.com/file/d/15GHFJnfcxrYNY8b56Lyxlz9Do29qaxIz/view?usp=sharing"
    file_id=df_propre_path.split('/')[-2]
    dwn_url='https://drive.google.com/uc?export=download&confirm=s5vl&id=' + file_id
    #st.write(dwn_url)
    df_propre = pd.read_csv(dwn_url)
    df_propre = df_propre.astype({"horodatage": "datetime64", 
                "Heure_estimee_de_passage_a_L_arret": "datetime64", 
                "date_heure" : "datetime64", 
                "date": "datetime64",
                "date_heure" : "datetime64",
                })

    df_meteo_path = "https://drive.google.com/file/d/1iW0786QIXpEDemgu70Z8w4DHmaSHCvPb/view?usp=sharing"
    file_id=df_meteo_path.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    #st.write(dwn_url)
    df_meteo = pd.read_csv(dwn_url, sep=";")

    df_arret_path = "https://drive.google.com/file/d/1oNZH4_I4mFZaPk1X75ddQdw72URfhAh4/view?usp=sharing"
    file_id=df_arret_path.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    #st.write(dwn_url)
    df_arret = pd.read_csv(dwn_url)

    df_result_ML_path = "https://drive.google.com/file/d/1fFexpM-bPXKifRQQdNseOikfuPN0tyLa/view?usp=sharing"
    file_id=df_result_ML_path.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    #st.write(dwn_url)
    df_result_ML = pd.read_csv(dwn_url)

    df_fichier_pour_pred_ML1_path = "https://drive.google.com/file/d/1nWa7KL5QVXccubvFHJe-HQ2n8XuJT6nL/view?usp=sharing"
    file_id=df_fichier_pour_pred_ML1_path.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    #st.write(dwn_url)
    df_fichier_pour_pred_ML1 = pd.read_csv(dwn_url, sep=";")

    return df_global, df_propre, df_meteo, df_arret, df_result_ML, df_fichier_pour_pred_ML1

def load_model():
    pickle_path = "Model_1_full_regressor.pkl"
    with open(pickle_path, 'rb') as file:  
        Pickled_LR_Model = pickle.load(file)
    return Pickled_LR_Model

def angers_bus_tram_IHM(): 
    st.set_page_config(layout="wide")
    
    df_global, df_propre, df_meteo, df_arret, df_result_ML, df_fichier_pour_pred_ML1= load_data()

    st.sidebar.title("Menu")
    menu = ["Accueil", "API & Features Engineering", "Analyse base consolidée", "Prediction de la qualité du trafic"]
    choix = st.sidebar.selectbox("Choix", menu)

    if choix == "Accueil":
        accueil()
    elif choix == "API & Features Engineering":
        api(df_global, df_meteo, )
    elif choix == "Analyse base consolidée":
        analyse_base_conso(df_propre, df_arret)
    elif choix == "Prediction de la qualité du trafic":
        prediction_qualite_traffic(df_propre, df_result_ML, df_fichier_pour_pred_ML1)
    

def accueil():
    st.header("Projet Simulation")
    st.markdown("> Clovis Delêtre & Charles Vitry")

    st.write("#### Description du projet")
    st.write("L'ambition de se projet est d'étudier la qualité du traffic des réseaux de transports en commun angevins et prévoir la qualité du trafic la veille pour le lendemain.")
    st.write("Pour cela, nous avons utilisé, dans un premier temps, les données de l'API de la ville d'Angers et ",
        "dans un second temps une base consolidées des requêtes de l'API.")
    st.write("Dans un premier temps, nous avons étudiés nos données dans l'objectif de faire resortir les informations importantes.")
    st.write("Dans un second temps, nous avons mis en place plusieurs modèles de machine learning pour prédire la qualité du traffic.")    


def api(df_global, df_meteo):
    df_api = request_api_tram()
    st.header("API & Features Engineering")
    st.write("### 1 - Récolte des données depuis l'API de la ville d'Angers ")
    st.write("A l'aide de la librairie requests, nous avons pu récupérer les données de l'API de la ville d'Angers sous forme de .json.")

    code_api = ''' 
            r = requests.get("https://data.angers.fr/api/records/1.0/search/",
            params = {
                "dataset":"bus-tram-position-tr",
                "rows":-1,
                },
            )
r.raise_for_status()
d = r.json()
    '''
    st.code(code_api, language='python')

    st.write(f"Actuellement il y a {df_api.shape[1]} informations différentes envoyés par les {df_api.shape[0]} bus et tramway de la ville d'Angers actuellement en service.")

    st.write("Avec l'aide du site de la ville d'Angers et de nos analyses on a les métas données suivantes :")

    col1, col2= st.columns(2)
    col1.write(">'record_timestamp'     = Horodatage")
    col1.write(">'fields.iddesserte'    = Identifiant SAE de désserte")
    col1.write(">'fields.mnemoligne'    = Mnemo de la ligne")
    col1.write(">'fields.numarret'      = Numero_Timeo_de_l_arret")
    col1.write(">'fields.etat'          = Etat SAE du véhicule")
    col1.write(">'fields.novh'          = N° de parc du véhicule")
    col1.write(">'fields.nomligne'      = Nom de la ligne")
    col1.write(">'fields.harret'        = Heure estimée de passage à L'arrêt")
    col1.write(">'fields.type'          = Modèle du véhicule")
    col1.write(">'fields.idligne'       = Identifiant SAE de ligne")
    col1.write(">'fields.ecart'         = Ecart horaire en secondes")
    col1.write(">'fields.dest'          = Destination")
    col1.write(">'fields.nomarret'      = Nom de l'arrêt")
    col2.write(">'fields.mnemoarret'    = Mne de l'arrêt")
    col2.write(">'fields.coordonnees'   = Coordonnées GPS WG84")
    col2.write(">'fields.idarret'       = Identifiant SAE de l'arrêt")
    col2.write(">'fields.cap'           = Cap du véhicule en degrés (gyromètre)")
    col2.write(">'fields.idparcours'    = Identifiant SAE du parcours")
    col2.write(">'fields.sv'            = Service voiture")
    col2.write(">'fields.y'             = Coordonnées GPS Lambert 2 Y")
    col2.write(">'fields.x'             = Coordonnées GPS Lambert 2 X")
    col2.write(">'fields.idvh'          = Identifiant du véhicule")
    col2.write(">'fields.ts_maj'        = Horodatage mise à jour")
    col2.write(">'geometry.coordinates' = Coordonnées")
    col2.write(">'horodatage_fichier'   = horodatage du fichier")
    col2.write(">'ecart_horodatage'     = ecart horodatage entre Horodatage et sa mise à jour")
    st.write("")

    st.button(label = "Voir les données", key = "button_api")
    if st.session_state.get("button_api"):
        st.dataframe(df_api)

    st.write("Pour la suite de l'étude, nous allons étudier une base consolidés des requêtes de l'API. Cette base a été créée à partir de 12 326 requêtes de l'API de la ville d'Angers.")

    st.write("### 2 - Features Engineering")

    st.write("#### 2.1 - Premier nettoyage des données")

    st.write("Dans un premier temps on supprime les colonnes : *fields.coordonnes, geometry.coordinates et geometry.type* qui n'ont pas d'importances ou repete les informations déjà présentes dans d'autres colonnes.")
    st.write("Par la suite on renomme les colonnes pour avoir des noms plus explicites en fonction des métadonnées.")
    


    st.write("#### 2.2 - Doublons / NA / valeurs aberrantes")

    st.write("Il important de supprimer les doublons, en effet, il peut y avoir plusieurs fois la même ligne dans le dataframe, cela peut être dû à une erreur de l'API ou à un problème de connexion.")
    code_doublons = '''
            duplicateRowsDF = df[df.drop(columns=["cordonnees_bus_geometrie", "coordonnees_GPS_WG84"]).duplicated()]
df = df.drop(duplicateRowsDF.index, axis=0)
    '''
    st.code(code_doublons, language='python')

    st.write("On regarde également le nombre de NA (valeurs manquantes), ici *15 628*, dans notre cas nous avons décidés de les supprimer.")
    code_na = '''
            df = df.dropna()
    '''
    st.code(code_na, language='python')

    st.write("Enfin, on regarde si il y a des valeurs aberrantes dans notre dataframe. Dans notre cas on a décider d'appliquer la méthode de *winsorisation* qui consiste à remplacer les valeurs aberrantes par les valeurs extrêmes.")
    code_aberrantes = '''
        from scipy.stats.mstats import winsorize
df["ecart_horaire_en_secondes"] = winsorize(df["ecart_horaire_en_secondes"], limits=[0.05, 0.05])
    '''
    st.code(code_aberrantes, language='python')

    #fig_avant_winsorizing, fig_apres_winsozing = graph_winsorizing(df_global)
    #col1, col2 = st.columns(2)
    #col1.plotly_chart(fig_avant_winsorizing)
    #col2.plotly_chart(fig_apres_winsozing)

    st.write("#### 2.3 - Developpement des données") 

    st.write("Dans un premier temps on a développé les coordonnées GPS en 2 colonnes : *latitude* et *longitude* pour faciler le traitement des informations géographiques par la suite.")
    code_developpement_gps = '''
    df[['latitude', 'longitude']] = df['coordonnees_GPS_WG84'].str.split(',', expand=True)
cols = ['latitude', 'longitude']
for col in cols :
    df[col] = df[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)
    '''
    st.code(code_developpement_gps, language='python')

    st.write("Dans un second temps on a divisé la colonne *horodatage* en plusieurs colonnes numériques qui vont être utile à l'affichage et à l'analyse du trafics. On récupère : ")
    st.write("- *year* : l'année")
    st.write("- *month* : le mois")
    st.write("- *day* : le jour")
    st.write("- *hours* : l'heure")
    st.write("- *minutes* : la minute")
    st.write("- *date* : nouveau format de la date")
    st.write("- *date_heure* : nouveau format de la date avec l'heure")
    st.write("- *jour_semaine* : le jour de la semaine")
    
    code_developpement_date = '''
    df["year"] = df["horodatage"].dt.year
df["month"] = df["horodatage"].dt.month
df["day"] =  df["horodatage"].dt.day
df["hours"] = df["horodatage"].dt.hour
df["minutes"] = df["horodatage"].dt.minute
df["date"] = pd.to_datetime(df[["year", "month", "day"]])
df["date_heure"] = pd.to_datetime(df[["year", "month", "day", "hours"]])

df["jour_semaine"] = df["date"].dt.day_name().map({"Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi", "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"})
    '''
    st.code(code_developpement_date, language='python')

    st.write("#### 2.4 - Ajout de nouvelles features")

    st.write("Pour perfectionner nos analyses et nos prédictions, l'idée d'aggréger de nouvelles features nous est venue.")
    st.write("Das un premier temps nous trouvé sur ce [site](https://www.historique-meteo.net/france/pays-de-la-loire/angers/2019/12/), l'historique météorologique de la ville d'Angers sur les dates de notre dataset.")
    st.write("On dispose des informations suivantes :")
    st.dataframe(df_meteo.head(5))

    st.write("Une autre idée a été de récupérer les informations liés aux arrêts de bus. Malheureusement la [source de données](https://data.angers.fr/explore/dataset/horaires-theoriques-et-arrets-du-reseau-irigo-gtfs/export/) a été mis à jour en 2021 mais nos données datent de 2019 donc ne sont pas utilisable.")

    st.write("#### 2.5 - Sauvegarde du dataframe")
    st.write("Avec l'ensemble de ses modifications, on arrive à un dataframe de *61 variables* pour *739 410 observations* près à l'utilisation.")

def analyse_base_conso(df_propre, df_arret):
    st.header("Analyse de la base consolidée")

    st.write("### 1 - Analyse générale")
    
    st.write("Premiere date : ", df_propre["horodatage"].min().strftime("%Y-%m-%d"))
    st.write("Derniere date : ", df_propre["horodatage"].max().strftime("%Y-%m-%d"))
    st.write("Nombre de jours étudiés : ", (df_propre["horodatage"].max() - df_propre["horodatage"].min()).days)
    st.write("Nombre de véhicules différents : ", df_propre["identifiant_du_vehicule"].nunique())
    st.write("Nombre de lignes différentes : ", df_propre["nom_de_la_ligne"].nunique())
    st.write("Nombre de données par mois : ")
    st.write(" - Aout 2019 : ", len(df_propre[df_propre["month"] == 8]))
    st.write(" - Septembre 2019 : ", len(df_propre[df_propre["month"] == 9]))
    st.write(" - Octobre 2019 : ", len(df_propre[df_propre["month"] == 10]))
    st.write(" - Novembre 2019 : ", len(df_propre[df_propre["month"] == 11]))
    st.write(" - Decembre 2019 : ", len(df_propre[df_propre["month"] == 12]))


    st.write("### 2 - Analyse des retards / avances => écart horaire en secondes")
    
    st.write("Globalement on remarque que quasiment 2/3 des véhicules sont en retard.")
    st.write("Ce qui reste cependant intriguant est le fait que 1/5 des bus soient en avance, ce qui ne devrait pas être le cas.")
    fig_pie_ecart = pie_ecart(df_propre)
    st.plotly_chart(fig_pie_ecart)

    col1, col2 = st.columns(2)

    fig_distribution_ecart = histo_distribution_ecart(df_propre)
    col1.plotly_chart(fig_distribution_ecart)

    col2.write("") 
    col2.write("") 
    col2.write("") 
    col2.write("") 
    col2.write("") 
    col2.write("") 
    col2.write("Si on s'intéresse aux distribution des écarts, on remarques plusieurs choses :")
    col2.write("> - la majorité des écarts sont faible (entre 0 et 19 secondes),")
    col2.write("> - En cas d'avance, leur valeur est faible (entre 0 et 140 secondes),")
    col2.write("> - Les retards, quand à eux sont plus variés et peuvent attendre des valeurs assez élevé comme on peut le voir avec la 'box-plot' de distribution des valeurs. ")

    df_mean_ecart = mean_ecart_value(df_propre)
    st.write("Globalement le retard moyens par jour est de ", round(df_mean_ecart["ecart_horaire_en_secondes"].mean(),2), " seconde ou ", round(df_mean_ecart["ecart_horaire_en_secondes"].mean()/60,2), " minutes.")

    st.write("Malgrés la différence en terme de nombre de données par mois, il est intéressant de voir l'évolution des écarts en fonctions des mois. ")
    st.write("On remarque que globalement les retards n'évoluent pas énormement enntre les mois de septembre / octobre / novembre mais que durant la période estivales, les écarts sont moindres. Ce phénomène peut s'expliquer par le fait que moins de bus circulent l'été. On peut égaler noter que par mois, les retard cummulés atteignent des valeurs importantes (~180k secondes soit ~50 heures de retard par mois).")
    fig_histo_month_ecart = histo_month_ecart(df_propre)
    st.plotly_chart(fig_histo_month_ecart)


    st.write("Si on avance dans le détails et qu'on s'intéresse aux écarts par jour et par mois on peut rermarquer plusieurs choses.")
    st.write(" - Rapidement on remarque un phénome de périodicité hebdomadaire avec une forte chute des écarts le dimanche et des pics les vendredis (début du weekend et période de forte affluence).")
    st.write(" - L'idée précedente qui différentie la période estivales des autres mois se retrouvent également ici.")
    st.write(" - Le mois de décembre étant incomplet il est difficile de conclure mais on peut remarquer que sur les premiers jours, les écarts semblent correspondre aux mois précédents.")
    st.write("*(Les deux graphes représentent la même information sous un format différent)*")
    col1, col2 = st.columns(2)
    fig_evo_ecart_month = evo_ecart_month(df_propre)
    fig_evo_ecart_month_v2 = evo_ecart_month_v2(df_propre)
    col1.plotly_chart(fig_evo_ecart_month)
    col2.plotly_chart(fig_evo_ecart_month_v2)

    st.write("Pour représenter les écarts par mois et par jour du mois, on peut utiliser une heatmap.")
    st.write("Nos précédentes hypothèses se vérifient ici (périodicité, différences entre la période estivale et la période scolaire et le manque d'informations sur la fin du mois de décembre).")
    fig_heatmap_ecart = heatmap_ecart(df_propre)
    st.plotly_chart(fig_heatmap_ecart)



    st.write("### 3 - Analyse différents modèles de véhicules")
    col1, col2 = st.columns(2)
    fig_repartion_tram_bus = repartion_tram_bus(df_propre)
    
    with col1:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("C'est évident, mais il y a beaucoup plus de bus que de tramway. Il y a ", 2134, " bus  uniques et " , 58, " tramways uniques.")

    col2.plotly_chart(fig_repartion_tram_bus)
    fig_histo_vehi_repartion = histo_vehi_repartion(df_propre)
    st.write("On remarque que certains modèles de véhicules sont plus utilisés que d'autres dans la ville d'Angers (ex : OMNICITY / OMNIART / GX327 / CITYSTD / CROSSWAY).")
    st.plotly_chart(fig_histo_vehi_repartion)
    


    st.write("### 4 - Analyse de l'état des véhicules")
    fig_histo_etat_vehicules = histo_etat_vehicules(df_propre)
    st.write("Ce qu'on peut déduire est une constence journalière (et hebdomadaire) du nombre de véhicules par état. Cependant la signification des différents états est difficile à trouver et donc à exploiter. Ce qu'on a pu trouver : ")
    st.write(" - 'HLP', course en Haut le pied (course sans client, entre dépôt et terminus)")
    st.write(" - 'TARR', Terminus arrivée ")
    st.write(" - 'TDEP' Terminus Départ ")
    st.write(" - 'LIGN' En ligne commerciale ")
    st.plotly_chart(fig_histo_etat_vehicules)


    st.write("### 5 - Analyse des lignes de bus / tram ")

    fig_pie_bus_tram = pie_bus_tram(df_propre)
    fig_evo_bus_tram = evo_bus_tram(df_propre)
    fig_heatmap_bus_tram = heatmap_bus_tram(df_propre)

    col1, col2 = st.columns(2)
    with col1:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("Comme on l'a vu précédemment, il y a plus de bus que de tramway. Ce phénomène se représente également sur le nombre de ligne de bus et le nombre de ligne de tram. Il y a ", 42, " bus  uniques et ", 2, " tramways uniques.")
    col2.plotly_chart(fig_pie_bus_tram)

    st.write("Observons l'évolution du nombre de bus / tramaway par ligne au cours du temps.")
    st.write(" A REVOIR")
    st.plotly_chart(fig_evo_bus_tram)

    st.write("On peut également représenter les écarts moyens (retards ou avances) par ligne en fonction du temps sous forme d'heatmap. L'idée est de répérer les lignes ou les mois qui cummulent le plus de retards moyens.")
    st.write(" - On remarque que les plus forts retards se retrouvent dans les *petites* lignes qui se dirigent vers les sorties d'Angers (ex lignes : 13 / 15 / 30 / 31 / 32 / 41 / 46).")
    st.write(" - On peut également noter que comme vu précédement les retards moyens sont plus faibles en période estivales.")
    st.plotly_chart(fig_heatmap_bus_tram, use_container_width = True)
    
    st.write("### 6 - Représentation géographique des données")

    st.write("On peut représenter les données sur une carte de la ville d'Angers. On peut ainsi voir les trajets des bus et tramway sur les différentes lignes en fonction du temps.")

    base_map = carte_bus_tram(df_propre, df_arret)
    
    _, col, _ = st.columns([1, 7, 1])
    with col : 
        folium_static(base_map, height = 800)





def prediction_qualite_traffic(df_propre, df_result_ML, df_fichier_pour_pred_ML1):
    st.header("Prédiction de la qualité du traffic")
    st.write("### 1 - Analyse du sujet ")
    st.write("La problèmatique à répondre étant : *Prévoir la qualité du trafic la veille pour le lendemain*.")

    st.write("Dans un premier temps il nous faut une variable qui va définir la qualité du traffic, dans notre cas nous avons décidé de prendre la variable *ecart_horaire_en_seconde*. En effet, s'il on prend la somme des écarts horaires sur une journée, on peut estimer si cette dernière va être considérée comme une bonne ou mauvaise journée en terme de retard.")

    st.write("Pour réaliser une étude du traffic toutes les variables présentées précédement ne sont pas 'utiles', de ce fait nous allons créer une base d'entrainement avec les variables qui nous semblent pertinentes et les regrouper par jour.")
    st.write("Les paramètres qui nous semblent importants sont : l'écart horaires, l'états des véhicules, les différentes lignes, la date (day, month, jour_semaine) et l'opinion météorologique")


    st.write("### 2 - Construction d'une base d'entrainement ")
    st.write("L'idée est de regrouper ses informations par jour, cela implique que les écarts vont être sommés et qu'on va dénombrer par jour le nombre de bus/trams par états et le nombre de bus/tram par ligne")

    st.write("L'ambition est d'avoir un jeu composé uniquement de valeurs numériques on va 'dummies' les variables jour_semaine & OPINION.") 
    code_merge_tab = """
    # Création de la base d'entrainement
df_date = df_pred[['date', 'month', 'day', 'jour_semaine', 'OPINION']].drop_duplicates()

# Dénombrement des états
df_etat_SAE_du_vehicule = df_pred[['date', 'etat_SAE_du_vehicule', 'identifiant_du_vehicule']].drop_duplicates()
df_etat_SAE_du_vehicule["count"] = 1
df_etat_SAE_du_vehicule = df_etat_SAE_du_vehicule.groupby(['date', 'etat_SAE_du_vehicule']).sum().reset_index().drop(columns=['identifiant_du_vehicule'])
df_etat_SAE_du_vehicule = df_etat_SAE_du_vehicule.pivot(index='date', columns='etat_SAE_du_vehicule', values='count').reset_index().fillna(0)

# Dénombrement par ligne
df_nom_de_la_ligne = df_pred[['date', 'nom_de_la_ligne', 'identifiant_du_vehicule']].drop_duplicates()
df_nom_de_la_ligne["count"] = 1
df_nom_de_la_ligne = df_nom_de_la_ligne.groupby(['date', 'nom_de_la_ligne']).sum().reset_index().drop(columns=['identifiant_du_vehicule'])
df_nom_de_la_ligne = df_nom_de_la_ligne.pivot(index='date', columns='nom_de_la_ligne', values='count').reset_index().fillna(0)

# Sum des écarts horaires
df_ecart = df_pred[['date', 'ecart_horaire_en_secondes']]
df_ecart = df_ecart.groupby(['date']).sum().reset_index()

# Merge des tables
df_pred = df_date.merge(df_etat_SAE_du_vehicule, on='date', how='left').merge(df_nom_de_la_ligne, on='date', how='left').merge(df_ecart, on='date', how='left').drop(columns=['date'])

# Dummies
df_pred = pd.get_dummies(df_pred, columns=['jour_semaine', 'OPINION'], drop_first=True)    
    """
    st.code(code_merge_tab, language = 'python')
    
    df_pred = create_base_entraiment(df_propre)
    # nbr row df_pred
    st.write("On arrive à un jeu d'entrainement de ",  df_pred.shape[0] ,"lignes et de ", df_pred.shape[1], "colonnes tel que :")
    st.dataframe(df_pred.head(5))

    st.write("### 3 - Création de plusieurs modèles de prédictions ")

    st.write("On va tester plusieurs modèles de prédictions pour voir lequel est le plus performant. L'idée étant de prédire une variable continue on va se focaliser sur des modèles de type *régression*. On va tester :")
    st.write("*L'ensemble des codes sont disponibles [ici](https://www.google.com/).*")
    st.write("- *LinearRegression*")
    st.write("- *Ridge*")
    st.write("- *RandomForestRegressor*")
    st.write("- *SGBRegressor*")
    st.write("- *MLPRegressor*")

    st.write("Pour chacun de ces modèles on va tester plusieurs paramètres pour voir lequel est le plus performant à l'aide d'une *GridSearchCV*.")

    st.write("Dans un premier temps on va séparer notre jeu d'entrainement en 2 parties. Une partie pour l'entrainement et une partie pour le test. On va utiliser la fonction *train_test_split* de *sklearn*.")
    st.write("Pour facilier le traitement des algos on va scaler nos données avec la fonction *StandardScaler* en se basant sur le jeu d'entrainement.")
    code_train_test_split = """
    from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

scalerx = StandardScaler()
scalery = StandardScaler()

X = df_pred.drop(columns=['ecart_horaire_en_secondes'])
y = df_pred['ecart_horaire_en_secondes']
y = np.array(y).reshape(-1,1)

scalerx.fit(X)
X = scalerx.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """
    st.code(code_train_test_split, language = 'python')
    
    st.write("Après entrainement de nos modèles on va comparer nos différents résultats avec plusieurs métrics : ")
    st.write("- *R2* : coefficient de détermination (ou coefficient de corrélation multiple) est une mesure de la qualité d'ajustement d'un modèle linéaire multiple.")
    st.latex(r'R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}')
    st.write("- *MSE* : Mean Squared Error (ou Erreur Quadratique Moyenne) est une mesure de la qualité d'ajustement d'un modèle linéaire multiple.")
    st.latex(r'MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2')
    st.write("- *MAE* : Mean Absolute Error (ou Erreur Absolue Moyenne) est une mesure de la qualité d'ajustement d'un modèle linéaire multiple.")
    st.latex(r'MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|')
    st.write("- *MAPE* : Mean Absolute Percentage Error (ou Erreur Absolue Moyenne en pourcentage) est une mesure de la qualité d'ajustement d'un modèle linéaire multiple.")
    st.latex(r'MAPE = \frac{1}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{y_i}')
    
    code_pickle = """
    import pickle
Pkl_Filename = "Model_1_full_regressor.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)
"""
    col1, col2 = st.columns([2,3])
    col1.write("")
    col1.write("")
    col1.write("D'après nos résultats on peut estimer que notre meilleur modèle est le *RandomForestRegressor*.")
    col1.write("Pour sauvegarder ce modèle on le sauvegarde son format pickle.")
    col1.code(code_pickle, language = 'python')
    col2.dataframe(df_result_ML)


    st.write("### 4 - Mise en pratique de notre modèle ")
    st.write("On va maintenant mettre en pratique notre modèle sur un jeu de données réel.")
    st.write("On va charger le jeu de données réel et on va le préparer de la même manière que le jeu d'entrainement.")
    st.write("On va ensuite prédire le retards cummulés sur la journée.")


    towrite = io.BytesIO()
    df_fichier_pour_pred_ML1.to_excel(towrite, index=False, encoding='utf-8')
    towrite.seek(0)
    
    col1, col2 = st.columns([2,2])
    col1.write("Dans un premier temps, il faut télécharger un fichier 'modèle' avec les informations nécéssaires à la prédiction.")
    col1.download_button(
        label="Télécharger le fichier csv",
        data=towrite,
        file_name='result_ML.xlsx',
        mime='text/csv',
        help = "Le fichier est pré-rempli avec les informations nécéssaires à la prédiction mais peuvent être modifiées."
    )
    col1.write("Quand le fichier est rempli il faut l'uploader dans l'application. (le drag and drop fonctionne)")

    df_test = None
    uploaded_files = col2.file_uploader("Choose a CSV file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        df_test = pd.read_excel(io.BytesIO(bytes_data))
        st.dataframe(df_test)

    if len(uploaded_files) > 0:
        jour_semaine = df_test['jour_semaine'][0]
        jour = df_test['day'][0]
        mois = df_test['month'][0]

        model = load_model()
        scalerx = get_scalerx(df_propre)
        df_test = pd.get_dummies(df_test)
        for col in ["jour_semaine_Lundi", "jour_semaine_Mardi", "jour_semaine_Mercredi", "jour_semaine_Jeudi", "jour_semaine_Vendredi", "jour_semaine_Samedi", "OPINION_météo défavorable", "OPINION_météo favorable", "OPINION_météo idéale", "OPINION_météo très défavorable"]:
            if col not in df_test.columns:
                df_test[col] = 0
        df_test = order_df(df_test)
        df_test = scalerx.transform(df_test)
        y_pred = model.predict(df_test)

        dict_mois = {1:"Janvier", 2:"Février", 3:"Mars", 4:"Avril", 5:"Mai", 6:"Juin", 7:"Juillet", 8:"Août", 9:"Septembre", 10:"Octobre", 11:"Novembre", 12:"Décembre"}
        mois = dict_mois[mois]
        st.write("D'après notre modèle le retard sur la journée du ", jour_semaine , " ", jour , " ", mois , " est de : ", y_pred[0], "secondes ou ", round(y_pred[0]/60,2), " minutes.")


    st.write("### 5 - Conclusion ")

    