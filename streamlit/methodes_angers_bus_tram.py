import pandas as pd
import requests
import datetime

from pandas.io.json import json_normalize

from scipy.stats.mstats import winsorize

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import folium
from folium import plugins
from folium.plugins import HeatMapWithTime
from folium.plugins import HeatMap

import streamlit as st

from sklearn.preprocessing import StandardScaler

def request_api_tram():
    r = requests.get("https://data.angers.fr/api/records/1.0/search/",
    params = {    
        "dataset":"bus-tram-position-tr",
        "rows":-1,    
        },
    )

    r.raise_for_status()
    
    d = r.json()

    df_api = pd.json_normalize(d['records'])

    return df_api 

@st.cache(suppress_st_warning=True)
def graph_winsorizing(df):
    df = (df
        .drop(columns=["Unnamed: 0", "datasetid", "recordid", "geometry.type"], errors='ignore')
        .astype({"record_timestamp": "datetime64", "fields.ts_maj": "datetime64", "fields.coordonnees": "string"})
        .assign(ecart_horodatage = lambda x: x["record_timestamp"] - x["fields.ts_maj"]))
    nomsColonnes = ['horodatage',
                'identifiant_SAE_de_desserte',
                'mnemo_de_la_ligne',
                'numero_Timeo_de_l_arret',
                'etat_SAE_du_vehicule',
                'numero_de_parc_du_vehicule',
                'nom_de_la_ligne',
                'Heure_estimee_de_passage_a_L_arret',
                'modele_du_vehicule',
                'identifiant_SAE_de_ligne',
                'ecart_horaire_en_secondes',
                'destination',
                'nom_de l_arret',
                'mne_de_l_arret',
                'coordonnees_GPS_WG84',
                'identifiant_SAE_de_l_arret',
                'cap_du_vehicule_en_degres',
                'identifiant_SAE_du_parcours',
                'service_voiture',
                'coordonnees_GPS_Y',
                'coordonnees_GPS_X',
                'identifiant_du_vehicule',
                'horodatage_maj',
                'cordonnees_bus_geometrie',
                'horodatage_fichier',
                'ecart_horodatage']
    df.columns = nomsColonnes
    duplicateRowsDF = df[df.drop(columns=["cordonnees_bus_geometrie", "coordonnees_GPS_WG84"]).duplicated()]
    df = df.drop(duplicateRowsDF.index, axis=0)
    df = df.dropna()
    fig = px.box(df, y="ecart_horaire_en_secondes", title="Ecart horaire en secondes (non winsorized)")
    df["ecart_horaire_en_secondes"] = winsorize(df["ecart_horaire_en_secondes"], limits=[0.05, 0.05])
    fig1 = px.box(df, y="ecart_horaire_en_secondes", title="Ecart horaire en secondes (winsorized)")
    return fig, fig1

@st.cache(suppress_st_warning=True)
def mean_ecart_value(df):
    df_mean_ecart = df[['date', 'ecart_horaire_en_secondes']]
    df_mean_ecart = df_mean_ecart.groupby(['date']).sum().reset_index()
    return df_mean_ecart

@st.cache(suppress_st_warning=True)
def pie_ecart(df):
    tab_ecart = [len(df[df["ecart_horaire_en_secondes"] > 0]), len(df[df["ecart_horaire_en_secondes"] < 0]) , len(df[df["ecart_horaire_en_secondes"] == 0])]
    tab_label = ["Vehicule en retard", "Véhicule en avance", "Véhicule à l'heure"]

    fig = go.Figure(data=[go.Pie(labels=tab_label, values=tab_ecart, pull = [0,0.2,0])])
    return fig

@st.cache(suppress_st_warning=True)
def histo_distribution_ecart(df):
    df_plot_ecart_horaire_en_seconde = df[['date_heure', 'ecart_horaire_en_secondes']]

    df_plot_ecart_horaire_en_seconde["count"] = 1
    df_plot_ecart_horaire_en_seconde = df_plot_ecart_horaire_en_seconde.groupby(['ecart_horaire_en_secondes']).sum().reset_index()

    fig = px.histogram(df_plot_ecart_horaire_en_seconde, x="ecart_horaire_en_secondes", y="count", title="Distribution des écarts horaire en secondes", nbins=50, marginal="box")
    return fig

@st.cache(suppress_st_warning=True)
def histo_month_ecart(df):
    ecart_jour = df.groupby(['month'])['ecart_horaire_en_secondes'].count()
    fig = px.bar(ecart_jour, 
                x=ecart_jour.index, 
                y='ecart_horaire_en_secondes', 
                title="Nombre de retard par mois en seconde",
                color='ecart_horaire_en_secondes')
    return fig

@st.cache(suppress_st_warning=True)
def evo_ecart_month(df):
    df_plot_ecart = df[['month', 'day', 'ecart_horaire_en_secondes']]
    df_plot_ecart = df_plot_ecart.groupby(['month', 'day']).sum().reset_index()

    df_plot_ecart_8 = df_plot_ecart[df_plot_ecart['month'] == 8]
    df_plot_ecart_9 = df_plot_ecart[df_plot_ecart['month'] == 9]
    df_plot_ecart_10 = df_plot_ecart[df_plot_ecart['month'] == 10]
    df_plot_ecart_11 = df_plot_ecart[df_plot_ecart['month'] == 11]
    df_plot_ecart_12 = df_plot_ecart[df_plot_ecart['month'] == 12]

    fig = make_subplots(rows=2, cols=3)
    fig.add_trace(go.Line(x=df_plot_ecart_8['day'], y=df_plot_ecart_8['ecart_horaire_en_secondes'], name = "Août"), row=1, col=1)
    fig.add_trace(go.Line(x=df_plot_ecart_9['day'], y=df_plot_ecart_9['ecart_horaire_en_secondes'], name = "Septembre"), row=1, col=2)
    fig.add_trace(go.Line(x=df_plot_ecart_10['day'], y=df_plot_ecart_10['ecart_horaire_en_secondes'], name = "Octobre"), row=1, col=3)
    fig.add_trace(go.Line(x=df_plot_ecart_11['day'], y=df_plot_ecart_11['ecart_horaire_en_secondes'], name = "Novembre"), row=2, col=1)
    fig.add_trace(go.Line(x=df_plot_ecart_12['day'], y=df_plot_ecart_12['ecart_horaire_en_secondes'], name = "Décembre"), row=2, col=2)
    fig.update_layout(xaxis_range=[0,31], yaxis_range=[0, 2500000])
    return fig

@st.cache(suppress_st_warning=True)
def evo_ecart_month_v2(df):
    df_plot_ecart = df[['month', 'day', 'ecart_horaire_en_secondes']]
    df_plot_ecart = df_plot_ecart.groupby(['month', 'day']).sum().reset_index()
    fig = px.bar(df_plot_ecart, 
            x='day',
            y='ecart_horaire_en_secondes',
            animation_frame="month",
            range_y=[0,2500000],
            )
    return fig

@st.cache(suppress_st_warning=True)
def heatmap_ecart(df):
    df_heatmap_ecart = df[['month', 'day', 'ecart_horaire_en_secondes']]
    df_heatmap_ecart = df_heatmap_ecart.groupby(['month', 'day']).sum().reset_index()
    df_heatmap_ecart = df_heatmap_ecart.pivot("month", "day", "ecart_horaire_en_secondes")
    df_heatmap_ecart = df_heatmap_ecart.fillna(0)
    df_heatmap_ecart.index = ["Août", "Septembre", "Octobre", "Novembre", "Décembre"]

    fig = px.imshow(df_heatmap_ecart, labels=dict(x="Jour", y="Mois", color="Ecart en seconde", height=1000, width=500, aspect="auto"))

    return fig

@st.cache(suppress_st_warning=True)
def repartion_tram_bus(df):
    df_plot_tram_bus = df[['identifiant_du_vehicule','mnemo_de_la_ligne']].drop_duplicates()
    df_plot_tram_bus["count"] = 1
    df_plot_tram_bus['type_ligne'] = df_plot_tram_bus['mnemo_de_la_ligne'].apply(lambda x: 'tram' if x[0] == 'A' else 'bus')
    df_plot_tram_bus_type = df_plot_tram_bus.groupby(['type_ligne']).sum().reset_index()

    tram_bus_value, tram_bus_labels = df_plot_tram_bus_type['count'].tolist(), df_plot_tram_bus_type['type_ligne'].tolist()
    fig = go.Figure(data=[go.Pie(labels=tram_bus_labels, values=tram_bus_value, hole=.3, pull=[0, 0.2])])
    return fig

@st.cache(suppress_st_warning=True)
def histo_vehi_repartion(df):
    df_plot_vehicule_repartition = df[['identifiant_du_vehicule', 'modele_du_vehicule']].drop_duplicates()
    df_plot_vehicule_repartition["count"] = 1
    df_plot_vehicule_repartition = df_plot_vehicule_repartition.groupby(['modele_du_vehicule']).sum().reset_index().drop(columns=["identifiant_du_vehicule"])

    fig = px.bar(df_plot_vehicule_repartition, 
                x='modele_du_vehicule', 
                y='count', 
                title="Répartition des différents modèles de véhicules",
                color='count')
    return fig

@st.cache(suppress_st_warning=True)
def histo_etat_vehicules(df):
    df_plot_vehicule_etat = df[['identifiant_du_vehicule', 'etat_SAE_du_vehicule', 'month', 'day']].drop_duplicates()
    df_plot_vehicule_etat["count"] = 1
    df_plot_vehicule_etat = df_plot_vehicule_etat.groupby(['month', 'day', 'etat_SAE_du_vehicule']).sum().reset_index().drop(columns=["identifiant_du_vehicule"])

    fig = px.bar(df_plot_vehicule_etat, x='day', y='count', animation_frame="month", color='etat_SAE_du_vehicule')
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def pie_bus_tram(df) :
    df_plot_tram_bus = df[['nom_de_la_ligne']].drop_duplicates()
    df_plot_tram_bus["count"] = 1
    df_plot_tram_bus['type_ligne'] = df_plot_tram_bus['nom_de_la_ligne'].apply(lambda x: 'tram' if x[0] == 'A' else 'bus')
    df_plot_tram_bus_type = df_plot_tram_bus.groupby(['type_ligne']).sum().reset_index()

    tram_bus_value, tram_bus_labels = df_plot_tram_bus_type['count'].tolist(), df_plot_tram_bus_type['type_ligne'].tolist()
    fig = go.Figure(data=[go.Pie(labels=tram_bus_labels, values=tram_bus_value, hole=.3, pull=[0, 0.2])])
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def evo_bus_tram(df):
    df_plot_bus_par_ligne = df[['date', 'month', 'identifiant_du_vehicule', 'mnemo_de_la_ligne']].drop_duplicates()
    df_plot_bus_par_ligne["count"] = 1
    df_plot_bus_par_ligne = df_plot_bus_par_ligne.groupby(['date', 'month', 'mnemo_de_la_ligne']).sum().reset_index().drop(columns=["identifiant_du_vehicule"])

    list_unique_mnemo_de_la_ligne = df_plot_bus_par_ligne.mnemo_de_la_ligne.unique().tolist()
    button_list = []
    for items in list_unique_mnemo_de_la_ligne:
        button_list.append(dict(label=items,
                                method="update",
                                args=[{"visible": [items in l for l in df_plot_bus_par_ligne["mnemo_de_la_ligne"]]},
                                    {"title": "Nombre de bus moyen par lignes : {}".format(items)}]))
    fig = px.line(df_plot_bus_par_ligne, x="date", y="count", color='mnemo_de_la_ligne', title="Nombre de bus moyen par lignes")
    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                buttons = button_list,
                direction = "down",
                pad={"r": 10, "t": 10},
                showactive = True
            )
        ]
    )
    return fig

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def heatmap_bus_tram(df):
    df_plot_ecart_horaire_par_ligne = df[['date_heure', 'month', 'identifiant_du_vehicule', 'mnemo_de_la_ligne', 'ecart_horaire_en_secondes']]
    df_plot_ecart_horaire_par_ligne_all = df_plot_ecart_horaire_par_ligne
    df_plot_ecart_horaire_par_ligne_all["count"] = 1
    df_plot_ecart_horaire_par_ligne_all = df_plot_ecart_horaire_par_ligne_all.groupby(['month', 'mnemo_de_la_ligne']).sum().reset_index().drop(columns=['identifiant_du_vehicule'])
    df_plot_ecart_horaire_par_ligne_all["moyenne_retard_seconde"] = round(df_plot_ecart_horaire_par_ligne_all["ecart_horaire_en_secondes"] / df_plot_ecart_horaire_par_ligne_all["count"],0)
    df_plot_ecart_horaire_par_ligne_all["moyenne_retard_minute"] = round(df_plot_ecart_horaire_par_ligne_all["moyenne_retard_seconde"] / 60, 0)

    df_heatmap_ecart_ligne = df_plot_ecart_horaire_par_ligne_all[["month", "mnemo_de_la_ligne", "moyenne_retard_seconde"]]
    df_heatmap_ecart_ligne = df_heatmap_ecart_ligne.pivot(index = "mnemo_de_la_ligne", columns = "month")["moyenne_retard_seconde"].fillna(0)

    fig = px.imshow(df_heatmap_ecart_ligne, height=800)
    return fig

def generateBaseMap(default_location=[47.478419, -0.563166], default_zoom_start=11):
    base_map = folium.Map(location=default_location, 
                    #tiles="Stamen Watercolor", 
                    control_scale=True, 
                    zoom_start=14)
    folium.TileLayer('openstreetmap').add_to(base_map)
    return base_map

#@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def carte_bus_tram(df, df_arret):
    df_map = df[['date_heure', 'latitude', 'longitude']]
    df_map["count"] = 1
    df_map['count'] = df_map[['date_heure', 'latitude', 'longitude', 'count']].groupby(['latitude', 'longitude', 'date_heure']).transform('count')

    df_map['date_heure']= pd.to_datetime(df_map['date_heure'])
    temps_index = []
    for i in df_map['date_heure'].unique():
        temps_index.append(i)
    date_labels = [pd.to_datetime(str(d)).strftime('%d/%m/%Y, %H') for d in temps_index]
    date_labels = [x for _,x in sorted(zip(temps_index,date_labels))]

    lat_long_list = []
    for i in df_map['date_heure'].unique():
        temp=[]
        for index, instance in df_map[df_map['date_heure'] == i].iterrows():
            temp.append([instance['latitude'],instance['longitude']])
        lat_long_list.append(temp)
    
    df_arret = df_arret[['stop_name', 'stop_lat', 'stop_lon']]
    # keep value with is inside df["mne_de_l_arret"] 
    df_arret = df_arret[df_arret['stop_name'].isin(df["mne_de_l_arret"])]

    base_map = generateBaseMap()
    cluster = plugins.MarkerCluster().add_to(base_map)

    for i in range (0, len(df_arret)):
        folium.Marker([df_arret.iloc[i]['stop_lat'], df_arret.iloc[i]['stop_lon']], popup=df_arret.iloc[i]['stop_name']).add_to(cluster)

    HeatMapWithTime(lat_long_list, radius=10, auto_play=True, position='bottomright', name="cluster", index=date_labels, max_opacity=0.9).add_to(base_map)

    return base_map


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def create_base_entraiment(df):
    df_pred = df[['ecart_horaire_en_secondes',
            'etat_SAE_du_vehicule', 
            'nom_de_la_ligne', 
            'identifiant_du_vehicule',
            'date',
            'month', 
            'day', 
            'hours', 
            'jour_semaine', 
            'OPINION'
            ]]
    df_date = df_pred[['date', 'month', 'day', 'jour_semaine', 'OPINION']].drop_duplicates()
    df_etat_SAE_du_vehicule = df_pred[['date', 'etat_SAE_du_vehicule', 'identifiant_du_vehicule']].drop_duplicates()
    df_etat_SAE_du_vehicule["count"] = 1
    df_etat_SAE_du_vehicule = df_etat_SAE_du_vehicule.groupby(['date', 'etat_SAE_du_vehicule']).sum().reset_index().drop(columns=['identifiant_du_vehicule'])
    df_etat_SAE_du_vehicule = df_etat_SAE_du_vehicule.pivot(index='date', columns='etat_SAE_du_vehicule', values='count').reset_index().fillna(0)
    df_nom_de_la_ligne = df_pred[['date', 'nom_de_la_ligne', 'identifiant_du_vehicule']].drop_duplicates()
    df_nom_de_la_ligne["count"] = 1
    df_nom_de_la_ligne = df_nom_de_la_ligne.groupby(['date', 'nom_de_la_ligne']).sum().reset_index().drop(columns=['identifiant_du_vehicule'])
    df_nom_de_la_ligne = df_nom_de_la_ligne.pivot(index='date', columns='nom_de_la_ligne', values='count').reset_index().fillna(0)
    df_ecart = df_pred[['date', 'ecart_horaire_en_secondes']]
    df_ecart = df_ecart.groupby(['date']).sum().reset_index()
    df_pred = df_date.merge(df_etat_SAE_du_vehicule, on='date', how='left').merge(df_nom_de_la_ligne, on='date', how='left').merge(df_ecart, on='date', how='left').drop(columns=['date'])
    df_pred = pd.get_dummies(df_pred, columns=['jour_semaine', 'OPINION'], drop_first=True)
    return df_pred

def get_scalerx(df):
    df_pred = create_base_entraiment(df)
    scalerx = StandardScaler()
    X = df_pred.drop(columns=['ecart_horaire_en_secondes'])
    scalerx.fit(X)
    return scalerx

def order_df(df):
    df = df[['month', 'day', 'DEV', 'DEVP', 'GARE', 'HC', 'HL', 'HLP', 'HLPR',
       'HLPS', 'HS', 'INC', 'LIGN', 'TARR', 'TDEP',
       'A - Remplacement Tram par Bus', 'ARDENNE <> ROSERAIE',
       'BEAUCOUZE <> ST BARTHELEMY', 'BELLE BEILLE <> MONPLAISIR',
       'BELLE BEILLE EXPRESS <> GARES', 'BOUCHEMAINE <> Z I  EST',
       'BRIOLLAY <> GARE', 'CIRCULAIRE VERNEAU GARE EUROPE', 'CORNE <> GARE',
       'CORNE <> GARE TRELAZE', 'D NAVETTE MARCHE MONPLAISIR',
       'DJF  BELLE BEILLE <> MONPLAISIR', 'DJF  TRELAZE <>  ST SYLVAIN',
       'DJF LORRAINE <> ST BARTHELEMY', 'DJF MURS ERIGNE <> MONPLAISIR',
       'DJF VILLAGE SANTE <> LORRAINE', 'ECOUFLANT GRIMORELLE <> GARE',
       'ECUILLE SOULAIRE <> GARE', 'ESPACE ANJOU <> EVENTARD',
       'EXPRESS CHANTOURTEAU <> GARES', 'EXPRESS MONTREUIL <> GARES',
       'FENEU CANTENAY <> GARE', 'HOPITAL <> MONTREUIL JUIGNE',
       'LA MEMBROLLE <> GARE', 'LAC MAINE <> STE GEMMES CL ANJOU',
       'M-MARCILLE <> ST AUBIN LA SALLE', 'MURS ERIGNE <> ADEZIERE SALETTE',
       'PLESSIS MACE MEIGNANNE <> GARE', 'PONTS CE <>  AQUAVITA H. RECULEE',
       'SARRIGNE PLESSIS <> GARE', 'SAVENNIERES <> GARE', 'SERVICE NUIT',
       'SOIR LAC MAINE <> CITE CHABADA', 'SOIR LORRAINE <> ST BARTH VERDUN',
       'SOIR MAIRIE PONTS CE <> AVRILLE', 'SOIR TRELAZE <>  LORRAINE',
       'SOUCELLES PELLOUAILLES <> GARE', 'SOULAINES <> GARE',
       'ST CLEMENT St LAMBERT <> GARE', 'ST LEGER St LAMBERT <> GARE',
       'ST LEZIN SORGES <> SCHWEITZER', 'ST MARTIN St JEAN <> GARE',
       'ST MATHURIN <> GARE', 'ST SYLVAIN BANCHAIS <>TRELAZE', 'jour_semaine_Jeudi', 'jour_semaine_Lundi',
       'jour_semaine_Mardi', 'jour_semaine_Mercredi', 'jour_semaine_Samedi',
       'jour_semaine_Vendredi', 'OPINION_météo défavorable',
       'OPINION_météo favorable', 'OPINION_météo idéale',
       'OPINION_météo très défavorable']]
    return df