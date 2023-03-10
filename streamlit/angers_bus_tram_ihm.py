import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from streamlit_folium import folium_static

import io
import pickle

from PIL import Image

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

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
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
                "horodatage_fichier": "datetime64",
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

    df_result_ML_2_path = "https://drive.google.com/file/d/19s6q9A6ETrVLQv2uwH-eiZ8rVn9x3xDG/view?usp=sharing"
    file_id=df_result_ML_2_path.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    #st.write(dwn_url)
    df_result_ML_2 = pd.read_csv(dwn_url)

    df_fichier_pour_pred_ML1_path = "https://drive.google.com/file/d/1nWa7KL5QVXccubvFHJe-HQ2n8XuJT6nL/view?usp=sharing"
    file_id=df_fichier_pour_pred_ML1_path.split('/')[-2]
    dwn_url='https://drive.google.com/uc?id=' + file_id
    #st.write(dwn_url)
    df_fichier_pour_pred_ML1 = pd.read_csv(dwn_url, sep=";")

    return df_global, df_propre, df_meteo, df_arret, df_result_ML, df_result_ML_2, df_fichier_pour_pred_ML1

def load_model():
    pickle_path = "Model_1_full_regressor.pkl"
    with open(pickle_path, 'rb') as file:  
        Pickled_LR_Model = pickle.load(file)
    return Pickled_LR_Model

def load_model2():
    pickle_path = "Model_2_full_regressor.pkl"
    with open(pickle_path, 'rb') as file:  
        Pickled_LR_Model2 = pickle.load(file)
    return Pickled_LR_Model2

def angers_bus_tram_IHM(): 
    st.set_page_config(layout="wide")
    
    df_global, df_propre, df_meteo, df_arret, df_result_ML, df_result_ML_2, df_fichier_pour_pred_ML1= load_data()

    st.sidebar.title("Menu")
    menu = ["Accueil", "API & Features Engineering", "Analyse base consolid??e", "Prediction de la qualit?? du trafic (par jour)", "Prediction de la qualit?? du trafic (par ligne)"]
    choix = st.sidebar.selectbox("Choix", menu)

    if choix == "Accueil":
        accueil()
    elif choix == "API & Features Engineering":
        api(df_global, df_meteo, df_propre)
    elif choix == "Analyse base consolid??e":
        analyse_base_conso(df_propre, df_arret)
    elif choix == "Prediction de la qualit?? du trafic (par jour)":
        prediction_qualite_traffic(df_propre, df_result_ML, df_fichier_pour_pred_ML1)
    elif choix == "Prediction de la qualit?? du trafic (par ligne)":
        prediction_qualite_traffic_par_ligne(df_propre, df_result_ML_2, df_fichier_pour_pred_ML1)

def accueil():
    st.header("Projet Simulation")
    st.markdown("> Clovis Del??tre & Charles Vitry")

    st.write("#### Description du projet")
    st.write("L'ambition de se projet est d'??tudier la qualit?? du traffic des r??seaux de transports en commun angevins et pr??voir la qualit?? du trafic la veille pour le lendemain.")
    st.write("Pour cela, nous avons utilis??, dans un premier temps, les donn??es de l'API de la ville d'Angers et ",
        "dans un second temps une base consolid??es des requ??tes de l'API.")
    st.write("Dans un premier temps, nous avons ??tudi??s nos donn??es dans l'objectif de faire resortir les informations importantes.")
    st.write("Dans un second temps, nous avons mis en place plusieurs mod??les de machine learning pour pr??dire la qualit?? du traffic.")    
    st.write("La source principale de donn??es est [ici](https://data.angers.fr/explore/dataset/bus-tram-position-tr/information/)")

def api(df_global, df_meteo, df_propre):
    df_api = request_api_tram()
    st.header("API & Features Engineering")
    st.write("### 1 - R??colte des donn??es depuis l'API de la ville d'Angers ")
    st.write("A l'aide de la librairie requests, nous avons pu r??cup??rer les donn??es de l'API de la ville d'Angers sous forme de .json.")

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

    st.write(f"Actuellement il y a {df_api.shape[1]} informations diff??rentes envoy??s par les {df_api.shape[0]} bus et tramway de la ville d'Angers actuellement en service.")

    st.write("Avec l'aide du site de la ville d'Angers et de nos analyses on a les m??tas donn??es suivantes :")

    col1, col2= st.columns(2)
    col1.write(">'record_timestamp'     = Horodatage")
    col1.write(">'fields.iddesserte'    = Identifiant SAE de d??sserte")
    col1.write(">'fields.mnemoligne'    = Mnemo de la ligne")
    col1.write(">'fields.numarret'      = Numero_Timeo_de_l_arret")
    col1.write(">'fields.etat'          = Etat SAE du v??hicule")
    col1.write(">'fields.novh'          = N?? de parc du v??hicule")
    col1.write(">'fields.nomligne'      = Nom de la ligne")
    col1.write(">'fields.harret'        = Heure estim??e de passage ?? L'arr??t")
    col1.write(">'fields.type'          = Mod??le du v??hicule")
    col1.write(">'fields.idligne'       = Identifiant SAE de ligne")
    col1.write(">'fields.ecart'         = Ecart horaire en secondes")
    col1.write(">'fields.dest'          = Destination")
    col1.write(">'fields.nomarret'      = Nom de l'arr??t")
    col2.write(">'fields.mnemoarret'    = Mne de l'arr??t")
    col2.write(">'fields.coordonnees'   = Coordonn??es GPS WG84")
    col2.write(">'fields.idarret'       = Identifiant SAE de l'arr??t")
    col2.write(">'fields.cap'           = Cap du v??hicule en degr??s (gyrom??tre)")
    col2.write(">'fields.idparcours'    = Identifiant SAE du parcours")
    col2.write(">'fields.sv'            = Service voiture")
    col2.write(">'fields.y'             = Coordonn??es GPS Lambert 2 Y")
    col2.write(">'fields.x'             = Coordonn??es GPS Lambert 2 X")
    col2.write(">'fields.idvh'          = Identifiant du v??hicule")
    col2.write(">'fields.ts_maj'        = Horodatage mise ?? jour")
    col2.write(">'geometry.coordinates' = Coordonn??es")
    col2.write(">'horodatage_fichier'   = horodatage du fichier")
    col2.write(">'ecart_horodatage'     = ecart horodatage entre Horodatage et sa mise ?? jour")
    st.write("")

    st.button(label = "Voir les donn??es", key = "button_api")
    if st.session_state.get("button_api"):
        st.dataframe(df_api)

    st.write("Pour la suite de l'??tude, nous allons ??tudier une base consolid??s des requ??tes de l'API. Cette base a ??t?? cr????e ?? partir de 12 326 requ??tes de l'API de la ville d'Angers.")

    st.write("### 2 - Features Engineering")

    st.write("#### 2.1 - Premier nettoyage des donn??es")

    st.write("Dans un premier temps on supprime les colonnes : *fields.coordonnes, geometry.coordinates et geometry.type* qui n'ont pas d'importances ou repete les informations d??j?? pr??sentes dans d'autres colonnes.")
    st.write("Par la suite on renomme les colonnes pour avoir des noms plus explicites en fonction des m??tadonn??es.")
    


    st.write("#### 2.2 - Doublons / NA / valeurs aberrantes")

    st.write("Il important de supprimer les doublons, en effet, il peut y avoir plusieurs fois la m??me ligne dans le dataframe, cela peut ??tre d?? ?? une erreur de l'API ou ?? un probl??me de connexion.")
    code_doublons = '''
            duplicateRowsDF = df[df.drop(columns=["cordonnees_bus_geometrie", "coordonnees_GPS_WG84"]).duplicated()]
df = df.drop(duplicateRowsDF.index, axis=0)
    '''
    st.code(code_doublons, language='python')

    st.write("On regarde ??galement le nombre de NA (valeurs manquantes), ici *15 628*, dans notre cas nous avons d??cid??s de les supprimer.")
    code_na = '''
            df = df.dropna()
    '''
    st.code(code_na, language='python')

    st.write("Enfin, on regarde si il y a des valeurs aberrantes dans notre dataframe. Dans notre cas on a d??cider d'appliquer la m??thode de *winsorisation* qui consiste ?? remplacer les valeurs aberrantes par les valeurs extr??mes.")
    code_aberrantes = '''
        from scipy.stats.mstats import winsorize
df["ecart_horaire_en_secondes"] = winsorize(df["ecart_horaire_en_secondes"], limits=[0.05, 0.05])
    '''
    st.code(code_aberrantes, language='python')

    #fig_avant_winsorizing, fig_apres_winsozing = graph_winsorizing(df_global)
    #col1, col2 = st.columns(2)
    #col1.plotly_chart(fig_avant_winsorizing)
    #col2.plotly_chart(fig_apres_winsozing)

    st.write("#### 2.3 - Developpement des donn??es") 

    st.write("Dans un premier temps on a d??velopp?? les coordonn??es GPS en 2 colonnes : *latitude* et *longitude* pour faciler le traitement des informations g??ographiques par la suite.")
    code_developpement_gps = '''
    df[['latitude', 'longitude']] = df['coordonnees_GPS_WG84'].str.split(',', expand=True)
cols = ['latitude', 'longitude']
for col in cols :
    df[col] = df[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)
    '''
    st.code(code_developpement_gps, language='python')

    st.write("Dans un second temps on a divis?? la colonne *horodatage* en plusieurs colonnes num??riques qui vont ??tre utile ?? l'affichage et ?? l'analyse du trafics. On r??cup??re : ")
    st.write("- *year* : l'ann??e")
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

    st.write("Pour perfectionner nos analyses et nos pr??dictions, l'id??e d'aggr??ger de nouvelles features nous est venue.")
    st.write("Das un premier temps nous trouv?? sur ce [site](https://www.historique-meteo.net/france/pays-de-la-loire/angers/2019/12/), l'historique m??t??orologique de la ville d'Angers sur les dates de notre dataset.")
    st.write("On dispose des informations suivantes :")
    st.dataframe(df_meteo.head(5))

    st.write("Une autre id??e a ??t?? de r??cup??rer les informations li??s aux arr??ts de bus. Malheureusement la [source de donn??es](https://data.angers.fr/explore/dataset/horaires-theoriques-et-arrets-du-reseau-irigo-gtfs/export/) a ??t?? mis ?? jour en 2021 mais nos donn??es datent de 2019 donc ne sont pas utilisable.")

    st.write("#### 2.5 - Sauvegarde du dataframe")
    st.write("Avec l'ensemble de ses modifications, on arrive ?? un dataframe de *61 variables* pour *739 410 observations* pr??s ?? l'utilisation.")

    st.write("#### 3 - Analyse du biais de nos donn??es")
    st.write("Examinons la fr??quence des requ??tes ?? la base de donn??es, et leurs d??calages avec l'envoie des informations du bus vers l'API.")

    fig_ecart_horodatage = ecart_horodatage(df_propre)
    st.plotly_chart(fig_ecart_horodatage)

    st.write("On remarque qu'il y a eu moins de requ??tes ?? l'API le samedi et le dimanche. Mais aussi une diff??rence de fr??quence de requ??te globale augment?? ?? partir de Septembre.")
    st.write("Cela peut ??tre expliqu?? de la mani??re suivante : pour ??conomiser du stockage, les requ??tes lors des p??riodes de haut trafic ont ??t?? privili??g??s.")
    st.write("En effet, si 100% des bus sont ?? l'heure le dimanche, la perception du retard ne sera quasiment pas impact??. tandis que si un bus contenant l'ensemble des voyageurs est en retard alors la perception du retard fortement impact??, m??me avec 99% des bus ?? l'heure.")

    st.write("Regardons les ??carts entre l'horodatage de l'API et l'horodatage du fichier")
    fig_ecart_horo_fichier = distrib_ecart_horo_fichier(df_propre)
    st.plotly_chart(fig_ecart_horo_fichier)
    st.write("Le nombre de lignes dont l'??cart est sup??rieur ?? 1h entre l'horodatage du bus et l'horodatage de la requetes ?? l'API est totalement n??gligable.")

def analyse_base_conso(df_propre, df_arret):
    st.header("Analyse de la base consolid??e")

    st.write("### 1 - Analyse g??n??rale")
    
    st.write("Premiere date : ", df_propre["horodatage"].min().strftime("%Y-%m-%d"))
    st.write("Derniere date : ", df_propre["horodatage"].max().strftime("%Y-%m-%d"))
    st.write("Nombre de jours ??tudi??s : ", (df_propre["horodatage"].max() - df_propre["horodatage"].min()).days)
    st.write("Nombre de v??hicules diff??rents : ", df_propre["identifiant_du_vehicule"].nunique())
    st.write("Nombre de lignes diff??rentes : ", df_propre["nom_de_la_ligne"].nunique())
    st.write("Nombre de donn??es par mois : ")
    st.write(" - Aout 2019 : ", len(df_propre[df_propre["month"] == 8]))
    st.write(" - Septembre 2019 : ", len(df_propre[df_propre["month"] == 9]))
    st.write(" - Octobre 2019 : ", len(df_propre[df_propre["month"] == 10]))
    st.write(" - Novembre 2019 : ", len(df_propre[df_propre["month"] == 11]))
    st.write(" - Decembre 2019 : ", len(df_propre[df_propre["month"] == 12]))


    st.write("### 2 - Analyse des retards / avances => ??cart horaire en secondes")
    
    st.write("Globalement on remarque que quasiment 2/3 des v??hicules sont en retard.")
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
    col2.write("Si on s'int??resse aux distribution des ??carts, on remarques plusieurs choses :")
    col2.write("> - la majorit?? des ??carts sont faible (entre 0 et 19 secondes),")
    col2.write("> - En cas d'avance, leur valeur est faible (entre 0 et 140 secondes),")
    col2.write("> - Les retards, quand ?? eux sont plus vari??s et peuvent attendre des valeurs assez ??lev?? comme on peut le voir avec la 'box-plot' de distribution des valeurs. ")

    df_mean_ecart = mean_ecart_value(df_propre)
    st.write("Globalement le retard moyens par jour est de ", round(df_mean_ecart["ecart_horaire_en_secondes"].mean(),2), " seconde ou ", round(df_mean_ecart["ecart_horaire_en_secondes"].mean()/60,2), " minutes.")

    st.write("Malgr??s la diff??rence en terme de nombre de donn??es par mois, il est int??ressant de voir l'??volution des ??carts en fonctions des mois. ")
    st.write("On remarque que globalement les retards n'??voluent pas ??normement enntre les mois de septembre / octobre / novembre mais que durant la p??riode estivales, les ??carts sont moindres. Ce ph??nom??ne peut s'expliquer par le fait que moins de bus circulent l'??t??. On peut ??galer noter que par mois, les retard cummul??s atteignent des valeurs importantes (~180k secondes soit ~50 heures de retard par mois).")
    fig_histo_month_ecart = histo_month_ecart(df_propre)
    st.plotly_chart(fig_histo_month_ecart)


    st.write("Si on avance dans le d??tails et qu'on s'int??resse aux ??carts par jour et par mois on peut rermarquer plusieurs choses.")
    st.write(" - Rapidement on remarque un ph??nome de p??riodicit?? hebdomadaire avec une forte chute des ??carts le dimanche et des pics les vendredis (d??but du weekend et p??riode de forte affluence).")
    st.write(" - L'id??e pr??cedente qui diff??rentie la p??riode estivales des autres mois se retrouvent ??galement ici.")
    st.write(" - Le mois de d??cembre ??tant incomplet il est difficile de conclure mais on peut remarquer que sur les premiers jours, les ??carts semblent correspondre aux mois pr??c??dents.")
    st.write("*(Les deux graphes repr??sentent la m??me information sous un format diff??rent)*")
    col1, col2 = st.columns(2)
    fig_evo_ecart_month = evo_ecart_month(df_propre)
    fig_evo_ecart_month_v2 = evo_ecart_month_v2(df_propre)
    col1.plotly_chart(fig_evo_ecart_month)
    col2.plotly_chart(fig_evo_ecart_month_v2)

    st.write("Pour repr??senter les ??carts par mois et par jour du mois, on peut utiliser une heatmap.")
    st.write("Nos pr??c??dentes hypoth??ses se v??rifient ici (p??riodicit??, diff??rences entre la p??riode estivale et la p??riode scolaire et le manque d'informations sur la fin du mois de d??cembre).")
    fig_heatmap_ecart = heatmap_ecart(df_propre)
    st.plotly_chart(fig_heatmap_ecart)

    st.write("En continuant dans l'analyse des ??carts, on peut s'int??resser ?? l'impact de la m??t??o sur les ??carts.")
    st.write("Avec ??vidence, on remarque que les jours avec une m??t??o tr??s d??favorable ou d??favorable ont des ??carts plus importants que les jours de 'beaux temps'.")
    fig_histo_impact_meteo = histo_impact_meteo(df_propre)
    st.plotly_chart(fig_histo_impact_meteo)


    st.write("### 3 - Analyse diff??rents mod??les de v??hicules")
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
        st.write("C'est ??vident, mais il y a beaucoup plus de bus que de tramway. Il y a ", 2134, " bus  uniques et " , 58, " tramways uniques.")

    col2.plotly_chart(fig_repartion_tram_bus)
    fig_histo_vehi_repartion = histo_vehi_repartion(df_propre)
    st.write("On remarque que certains mod??les de v??hicules sont plus utilis??s que d'autres dans la ville d'Angers (ex : OMNICITY / OMNIART / GX327 / CITYSTD / CROSSWAY).")
    st.plotly_chart(fig_histo_vehi_repartion)
    


    st.write("### 4 - Analyse de l'??tat des v??hicules")
    fig_histo_etat_vehicules = histo_etat_vehicules(df_propre)
    st.write("Ce qu'on peut d??duire est une constence journali??re (et hebdomadaire) du nombre de v??hicules par ??tat. Cependant la signification des diff??rents ??tats est difficile ?? trouver et donc ?? exploiter. Ce qu'on a pu trouver : ")
    st.write(" - 'HLP', course en Haut le pied (course sans client, entre d??p??t et terminus)")
    st.write(" - 'TARR', Terminus arriv??e ")
    st.write(" - 'TDEP' Terminus D??part ")
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
        st.write("Comme on l'a vu pr??c??demment, il y a plus de bus que de tramway. Ce ph??nom??ne se repr??sente ??galement sur le nombre de ligne de bus et le nombre de ligne de tram. Il y a ", 42, " bus  uniques et ", 2, " tramways uniques.")
    col2.plotly_chart(fig_pie_bus_tram)

    #st.write("Observons l'??volution du nombre de bus / tramaway par ligne au cours du temps.")
    #st.write(" A REVOIR")
    #st.plotly_chart(fig_evo_bus_tram)

    st.write("On peut ??galement repr??senter les ??carts moyens (retards ou avances) par ligne en fonction du temps sous forme d'heatmap. L'id??e est de r??p??rer les lignes ou les mois qui cummulent le plus de retards moyens.")
    st.write(" - On remarque que les plus forts retards se retrouvent dans les *petites* lignes qui se dirigent vers les sorties d'Angers (ex lignes : 13 / 15 / 30 / 31 / 32 / 41 / 46).")
    st.write(" - On peut ??galement noter que comme vu pr??c??dement les retards moyens sont plus faibles en p??riode estivales.")
    st.plotly_chart(fig_heatmap_bus_tram, use_container_width = True)
    
    st.write("### 6 - Repr??sentation g??ographique des donn??es")

    st.write("On peut repr??senter les donn??es sur une carte de la ville d'Angers. On peut ainsi voir les trajets des bus et tramway sur les diff??rentes lignes en fonction du temps.")

    base_map = carte_bus_tram(df_propre, df_arret)
    
    _, col, _ = st.columns([1, 7, 1])
    with col : 
        folium_static(base_map, height = 800)





def prediction_qualite_traffic(df_propre, df_result_ML, df_fichier_pour_pred_ML1):
    st.header("Pr??diction de la qualit?? du traffic")
    st.write("### 1 - Analyse du sujet ")
    st.write("La probl??matique ?? r??pondre ??tant : *Pr??voir la qualit?? du trafic la veille pour le lendemain*.")

    st.write("Dans un premier temps il nous faut une variable qui va d??finir la qualit?? du traffic, dans notre cas nous avons d??cid?? de prendre la variable *ecart_horaire_en_seconde*. En effet, s'il on prend la somme des ??carts horaires sur une journ??e, on peut estimer si cette derni??re va ??tre consid??r??e comme une bonne ou mauvaise journ??e en terme de retard.")

    st.write("Pour r??aliser une ??tude du traffic toutes les variables pr??sent??es pr??c??dement ne sont pas 'utiles', de ce fait nous allons cr??er une base d'entrainement avec les variables qui nous semblent pertinentes et les regrouper par jour.")
    st.write("Les param??tres qui nous semblent importants sont : l'??cart horaires, l'??tats des v??hicules, les diff??rentes lignes, la date (day, month, jour_semaine) et l'opinion m??t??orologique")


    st.write("### 2 - Construction d'une base d'entrainement ")
    st.write("L'id??e est de regrouper ses informations par jour, cela implique que les ??carts vont ??tre somm??s et qu'on va d??nombrer par jour le nombre de bus/trams par ??tats et le nombre de bus/tram par ligne")

    st.write("L'ambition est d'avoir un jeu compos?? uniquement de valeurs num??riques on va encoder les variables jour_semaine & OPINION.") 

    st.write("De plus, l'id??e ??tant de pr??dire les valeurs la veille pour le lendemain, on va shift nos donn??es de 1 (par date) pour avoir entrainer un mod??le sur les valeurs de la veille. De ce fait, en utilisant les valeurs de la veille, on peut pr??dire le retard du lendemain")
    code_merge_tab = """
    # Cr??ation de la base d'entrainement
df_date = df_pred[['date', 'month', 'day', 'jour_semaine', 'OPINION']].drop_duplicates()

# D??nombrement des ??tats
df_etat_SAE_du_vehicule = df_pred[['date', 'etat_SAE_du_vehicule', 'identifiant_du_vehicule']].drop_duplicates()
df_etat_SAE_du_vehicule["count"] = 1
df_etat_SAE_du_vehicule = df_etat_SAE_du_vehicule.groupby(['date', 'etat_SAE_du_vehicule']).sum().reset_index().drop(columns=['identifiant_du_vehicule'])
df_etat_SAE_du_vehicule = df_etat_SAE_du_vehicule.pivot(index='date', columns='etat_SAE_du_vehicule', values='count').reset_index().fillna(0)

# D??nombrement par ligne
df_nom_de_la_ligne = df_pred[['date', 'nom_de_la_ligne', 'identifiant_du_vehicule']].drop_duplicates()
df_nom_de_la_ligne["count"] = 1
df_nom_de_la_ligne = df_nom_de_la_ligne.groupby(['date', 'nom_de_la_ligne']).sum().reset_index().drop(columns=['identifiant_du_vehicule'])
df_nom_de_la_ligne = df_nom_de_la_ligne.pivot(index='date', columns='nom_de_la_ligne', values='count').reset_index().fillna(0)

# Sum des ??carts horaires
df_ecart = df_pred[['date', 'ecart_horaire_en_secondes']]
df_ecart = df_ecart.groupby(['date']).sum().reset_index()

# Merge des tables
df_pred = df_date.merge(df_etat_SAE_du_vehicule, on='date', how='left').merge(df_nom_de_la_ligne, on='date', how='left').merge(df_ecart, on='date', how='left').drop(columns=['date'])

# Dummies
df_pred = pd.get_dummies(df_pred, columns=['jour_semaine', 'OPINION'], drop_first=True)

# Shift de 1
mask = ~(df_pred.columns.isin(['month','day']))
cols_to_shift = df_pred.columns[mask]
df_pred[cols_to_shift] = df_pred.loc[:,mask].shift(1)
df_pred = df_pred.dropna()
    """
    st.code(code_merge_tab, language = 'python')
    
    df_pred = create_base_entraiment(df_propre)
    # nbr row df_pred
    st.write("On arrive ?? un jeu d'entrainement de ",  df_pred.shape[0] ,"lignes et de ", df_pred.shape[1], "colonnes tel que :")
    st.dataframe(df_pred.head(5))

    st.write("### 3 - Cr??ation de plusieurs mod??les de pr??dictions ")

    st.write("On va tester plusieurs mod??les de pr??dictions pour voir lequel est le plus performant. L'id??e ??tant de pr??dire une variable continue on va se focaliser sur des mod??les de type *r??gression*. On va tester :")
    
    st.write("L'utilisation de la classification ne permet pas de quantifier, or le temps se doit d'??tre quantifiable, ex : 'votre avion a un retard mod??r??' ne permet pas de quantification et donc pas de  prise de d??cision sur le choix d'un autre mode de transport .")
    st.write("Utilis?? la classification pour notre probl??matique desservirait l'objectif d'un retard per??u moindre, et ce par la perte de variabilit?? engendr??.")
    st.write("On peut n??anmoins song?? ?? un algorithme de CAH pour labellis?? les retards pr??vu")

    st.write("*L'ensemble des codes sont disponibles [ici](https://github.com/ClovisDel/qualite_trafic_tram_bus_angers).*")
    st.write("- *LinearRegression*")
    st.write("- *Ridge*")
    st.write("- *RandomForestRegressor*")
    st.write("- *SGBRegressor*")
    st.write("- *MLPRegressor*")

    st.write("Pour chacun de ces mod??les on va tester plusieurs param??tres pour voir lequel est le plus performant ?? l'aide d'une *GridSearchCV*.")

    st.write("Dans un premier temps on va s??parer notre jeu d'entrainement en 2 parties. Une partie pour l'entrainement et une partie pour le test. On va utiliser la fonction *train_test_split* de *sklearn*.")
    st.write("Pour facilier le traitement des algos on va scaler nos donn??es avec la fonction *StandardScaler* en se basant sur le jeu d'entrainement.")
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
    
    st.write("Apr??s entrainement de nos mod??les on va comparer nos diff??rents r??sultats avec plusieurs m??trics : ")
    st.write("- *R2* : coefficient de d??termination (ou coefficient de corr??lation multiple) est une mesure de la qualit?? d'ajustement d'un mod??le lin??aire multiple.")
    st.latex(r'R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}')
    st.write("- *MSE* : Mean Squared Error (ou Erreur Quadratique Moyenne) est une mesure de la qualit?? d'ajustement d'un mod??le lin??aire multiple.")
    st.latex(r'MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2')
    st.write("- *MAE* : Mean Absolute Error (ou Erreur Absolue Moyenne) est une mesure de la qualit?? d'ajustement d'un mod??le lin??aire multiple.")
    st.latex(r'MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|')
    st.write("- *MAPE* : Mean Absolute Percentage Error (ou Erreur Absolue Moyenne en pourcentage) est une mesure de la qualit?? d'ajustement d'un mod??le lin??aire multiple.")
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
    col1.write("D'apr??s nos r??sultats on peut estimer que notre meilleur mod??le est le *RandomForestRegressor*.")
    col1.write("Pour sauvegarder ce mod??le on le sauvegarde son format pickle.")
    col1.code(code_pickle, language = 'python')
    col2.dataframe(df_result_ML)


    st.write("### 4 - Mise en pratique de notre mod??le ")
    st.write("On va maintenant mettre en pratique notre mod??le sur un jeu de donn??es r??el.")
    st.write("On va charger le jeu de donn??es r??el et on va le pr??parer de la m??me mani??re que le jeu d'entrainement.")
    st.write("On va ensuite pr??dire le retards cummul??s sur la journ??e.")


    towrite = io.BytesIO()
    df_fichier_pour_pred_ML1.to_excel(towrite, index=False, encoding='utf-8')
    towrite.seek(0)
    
    col1, col2 = st.columns([2,2])
    col1.write("Dans un premier temps, il faut t??l??charger un fichier 'mod??le' avec les informations n??c??ssaires ?? la pr??diction.")
    col1.download_button(
        label="T??l??charger le fichier csv",
        data=towrite,
        file_name='result_ML.xlsx',
        mime='text/csv',
        help = "Le fichier est pr??-rempli avec les informations n??c??ssaires ?? la pr??diction mais peuvent ??tre modifi??es."
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
        for col in ["jour_semaine_Lundi", "jour_semaine_Mardi", "jour_semaine_Mercredi", "jour_semaine_Jeudi", "jour_semaine_Vendredi", "jour_semaine_Samedi", "OPINION_m??t??o d??favorable", "OPINION_m??t??o favorable", "OPINION_m??t??o id??ale", "OPINION_m??t??o tr??s d??favorable"]:
            if col not in df_test.columns:
                df_test[col] = 0
        df_test = order_df(df_test)
        df_test = scalerx.transform(df_test)
        y_pred = model.predict(df_test)

        dict_mois = {1:"Janvier", 2:"F??vrier", 3:"Mars", 4:"Avril", 5:"Mai", 6:"Juin", 7:"Juillet", 8:"Ao??t", 9:"Septembre", 10:"Octobre", 11:"Novembre", 12:"D??cembre"}
        mois = dict_mois[mois]
        st.write("D'apr??s notre mod??le le retard sur la journ??e du ", jour_semaine , " ", jour , " ", mois , " est de : ", y_pred[0], "secondes ou ", round(y_pred[0]/60,2), " minutes.")


def prediction_qualite_traffic_par_ligne(df_propre, df_result_ML_2, df_fichier_pour_pred_ML1):
    st.header("Pr??diction de la qualit?? du traffic")
    st.write("### 1 - Analyse du sujet ")
    st.write("En prenant le m??me probl??me pr??c??dant, une autre analyse possible est de se positionner en tant qu'utilisateur de ses donn??es. G??n??ralement, un utilisateur utilise toujours la m??me ligne de tram ou de bus, qu'il utilise pour faire l'aller puis le retour.")
    st.write("Nous r??aliserons donc des **pr??visions par lignes de bus/tram**, puisque qu'une pr??diction globale n'a que peu d'int??r??t pour l'utilisateur et qu'une pr??diction plus pr??cise n??cessite plus de donn??es.")
    st.write("Pour symboliser la qualit?? du trafic, nous allons utiliser la variable *ecart_horaire_en_secondes* qui est une variable num??rique. Cela implique d'utiliser un mod??le de r??gression.")
    st.write("Pour pouvoir agr??ger ces donn??es par jours, nous choisissons l'agr??gation de somme, cela permettra de mettre en avant la variabilit?? entre les lignes. Puisque plus une ligne est utilis??e, plus le nombre de bus ?? son service est haut et donc plus de requ??tes qui la concerne sont effectu??es. Ainsi si une ligne est tr??s utilis??e et a de nombreux retards alors sa somme des ??carts sera tr??s importante. Cela permettra de p??naliser ?? hauteur de la fr??quentation de la ligne. (Selon l???hypoth??se bcp de voyageurs => bcp de bus en activit?? sur cette ligne)") 
    st.write("R??sum?? : en entr??e nous aurons les donn??es agr??g??es de la veille en globale et les derni??res donn??es agr??g??es par ligne, ainsi que la ligne que nous souhaitons pr??dire, en sortie nous aurons la somme des ??carts horaires de la ligne que nous souhaitons pr??dire.")    

    try :
        image = Image.open('model_fonctionnement.png')
        st.image(image, caption='model_fonctionnement')
    except : 
        pass

    st.write("### 2 - Pr??paration des donn??es ")

    code_prep_data = """
    # Si le bus/tram est arriv?? en avance, on consid??re que l'??cart horaire est nul 
# => ??viter que les bus en avance annule ceux en retards lors de l'op??ration de somme
df_RLM['ecart_horaire_en_secondes'] = df_RLM['ecart_horaire_en_secondes'].apply(lambda x: 0 if x < 0 else x)

# Aggr??gation par jour et par ligne
df_group = df_RLM.groupby(['date','nom_de_la_ligne']).agg(
    {'ecart_horaire_en_secondes' : 'sum',
     'diff_estimee' : 'sum',
     'diff_maj' : 'sum',
     'month' : 'first',
     'day' : 'first',
     'jour_semaine' : 'first',
     'OPINION' : 'first'
    }).merge((df_RLM
  .groupby(["date",'nom_de_la_ligne', 'etat_SAE_du_vehicule'])
  .size()
  .unstack('etat_SAE_du_vehicule', fill_value=0)
  .add_prefix("nombre_etat_")
), on=['date','nom_de_la_ligne'], how='left')
     		
# Les valeurs d'??carts prochaines pour chaque ligne et la date de celles-ci
df_group[['next_ecart','month','day','jour_semaine']] = df_group.groupby('nom_de_la_ligne')[['ecart_horaire_en_secondes','month','day','jour_semaine']].shift()
df_group.dropna(inplace=True)       
  
df_group.reset_index(inplace=True, level=['nom_de_la_ligne'])
  
#Ajout donn??es globales de la veille  
df_group.merge(
  df_RLM.groupby('date').agg({
    'ecart_horaire_en_secondes' : 'sum',
     'diff_estimee' : 'sum',
     'diff_maj' : 'sum',  }).rename(columns=
                  {'ecart_horaire_en_secondes': 'ecart_horaire_en_secondes_global',
                   'diff_estimee' : 'diff_estimee_global',
                   'diff_maj' : 'diff_maj_global'}), on=['date'], how='left')
df_group = pd.get_dummies(df_group, columns=['OPINION','nom_de_la_ligne'])
"""
    st.code(code_prep_data, language='python')

    st.write("On remarque que certains jours, il n'y a que tr??s peu de lignes de bus actives. et certaines lignes de bus n'ont que tr??s peu de donn??es. On pourrait limiter notre pr??diction aux lignes les plus actives.")
    st.write("Mais cela p??naliserait les utilisateurs de petite ligne, pour r??gler cela nous avons r??aliser le shift sur les dates en plus de la valeur ?? pr??dire. Cela permet de r??cup??rer l'information de saisonnalit?? perdu lorsqu'une ligne n'est pas active chaque jour.")

    st.write("### 3 - Mod??lisation par le mod??le Prophet")
    st.write("Dans un premier temps, essayons de pr??dire la variable expliqu?? avec uniquement la variable date. Pour cela, nous allons utiliser le mod??le Prophet de Facebook. Ce mod??le est bas?? sur la d??composition de la s??rie temporelle en trois composantes : tendance, saisonnalit?? et bruit. Il est donc particuli??rement adapt?? ?? la pr??diction de s??ries temporelles ?? condition qu'il y ait un lien entre la variable ?? expliqu?? et la date.")

    code_prophet = """
    from prophet import Prophet
m = Prophet()

df_prophet = df_group['next_ecart'].reset_index()

df_prophet = df_prophet.rename(columns={'date': 'ds', 'next_ecart': 'y'})
train , test = train_test_split(df_prophet, test_size=0.2, random_state=0)
m.fit(train)
"""
    st.code(code_prophet, language='python')

    st.write("En comparant les r??sultats de la pr??diction par la moyenne et par le mod??le Prophet, on peut voir que le mod??le Prophet a des performances exc??crables.") 
    st.write("Cela est d?? au fait que l'horodatage ne contient peu ou pas d'information sur la variable ?? expliqu?? ou que la connaissance n'est disponible que par combinaisons avec d'autres variables.")
    st.write("Nous pourions utiliser le mod??le Prophet pour expliquer la saisonnalit?? et la tendance, puis entrainer des mod??les de r??gression sur les r??sidus. Cependant, il s'agit d'un mod??le inutilement complexe pour le nombre d'observations limit??s que nous avons.")
    st.write("Continuons dans les mod??les explicables avec une r??gression lin??aire multiple qui implique un lien lin??aire entre la variable ?? expliqu?? et les variables explicatives.")

    st.write("### 4 - Analyse des liens entre nos variables")
    st.write("Avant d'utiliser nos mod??les vu pr??c??dement nous allons essayer de tirer les liens polynomiaux entre nos variables")
    code_poly = """
    from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
poly.fit(Xtrain)
Xtrain_poly = poly.transform(Xtrain)
Xtest_poly = poly.transform(Xtest)
Xtrain_poly.shape
"""
    st.code(code_poly, language='python')
    st.write("Cependant, il en ressort", 2347 ,"collones ?? cause de nos variables qualitative mis en one-hot-encoding.")
    st.write("Ce qui d??truirait l'explicabilit??, alors que cela est l'objectif d'avoir un mod??le simple.")

    st.write("### 5 - Mod??lisation par la r??gression lin??aire multiple")
    st.write("Suite ?? cette mod??lisation, nous appliquons les mod??les vu pr??c??dement sur nos donn??es.")
    st.write("*L'ensemble des codes sont disponibles [ici](https://github.com/ClovisDel/qualite_trafic_tram_bus_angers).*")
    st.write("On obtient les r??sultats suivant") 

    df_result_ML_2["MAPE"] = df_result_ML_2["MAPE"].apply(lambda x: '%.2E' % x)
    df_result_ML_2["MSE"] = df_result_ML_2["MSE"].apply(lambda x: '%.2E' % x)
    df_result_ML_2["MAE"] = df_result_ML_2["MAE"].apply(lambda x: '%.2E' % x)
    _, col, _ = st.columns([1, 3, 1])
    col.dataframe(df_result_ML_2)

    #model2 = load_model2()

    st.write("### 5 - Conclusion")

    st.write("Nous avons pu cr??er un mod??le qui pr??dis le trafic le lendemain en exprimant 92% de la variabilit?? et 89% si l???on s??lectionne la ligne ?? pr??dire, ces mod??les sont respectivement 7 et 4.5 fois meilleur que la pr??diction par la moyenne en terme de MAE.")
    st.write("Mais nous ne pouvons ignorer ses biais puisqu???il n???a ??t?? entrain?? que sur 128 jours. Cela nous a permis de nous rendre compte de l???importance du contr??le de la donn??e, qu???il ne suffisait pas d???avoir des centaines de milliers de lignes pour abattre n???importe quelle probl??matique. ")
    st.write("De plus, ce projet a permis de consolider nos connaissances en data science, et nous avons d??j?? pu mettre en application nos nouvelles comp??tences en data visualisation dans nos entreprises respectives avec succ??s.")