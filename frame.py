# cd my file path
# $ streamlit run frame.py
# save and refresh the webpage whenever there's any change in codes

import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import datetime as dt
import plotly.express as px
import os
import pydeck as pdk
# from mpl_toolkits.basemap import Basemap

# def _max_width_():
#     max_width_str = f"max-width: 1200px;"
#     st.markdown(
#         f"""
#     <style>
#     .reportview-container .main .block-container{{
#         {max_width_str}
#     }}
#     </style>    
#     """,
#         unsafe_allow_html=True,
#     )

# _max_width_()

# Main title
# use st.header / st.subheader for section titles
st.title("U.S. Protests for Racial Justice")
st.write("Kyle Skene, Houjiang Liu, Yuan Hua")

df = pd.read_excel("data/protest.xlsx")
df = df.fillna(0)
df = df.drop(columns=['Unnamed: 0','Country', 'EstimateText', 'AdjustedLow', 'Actor', 'Claim', 'Pro(2)/Anti(1)', 'TownsCities', 'Events', 'Misc.', 'BestGuess','Source1','Source2','Source3'])

df = df.fillna(0)

df['EstimateLow'].loc[df['EstimateLow'] == "200-300"] = 300
df['EstimateHigh'].loc[df['EstimateHigh'] == "200-300"] = 300

df['ReportedArrests'].loc[df['ReportedArrests'] == 'numerous '] = 0
df['ReportedArrests'].loc[df['ReportedArrests'] == 'several '] = 0
df['ReportedArrests'].loc[df['ReportedArrests'] == 'about a dozen'] = 0
df['ReportedArrests'].loc[df['ReportedArrests'] == 'numerous'] = 0

df['EstimateLow'] = pd.to_numeric(df['EstimateLow'])
df['EstimateHigh'] = pd.to_numeric(df['EstimateHigh'])
df['ReportedArrests'] = pd.to_numeric(df['ReportedArrests'])

df = df.rename(columns = {'state_name':'State_name'})

# st.write(df)

# Show our data
# st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
# st.header("Data")
# st.dataframe(df.head())


## show each protest on the map (4017 rows of data)
st.subheader("Protests Distribution: May25 – Jun 28")
st.write("*Each dot represents one protest.*") # Italics

df3 = df
df3["place"] = df3["City"] + [','] + df3["State_id"]
df3 = df3[["place","lat","lng","Date"]]

df3['lng'] = df3['lng'].astype('str')
df3['lng'] = df3['lng'].str.replace(',', '')
df3['lng'] = df3['lng'].astype('float')
df3['lat'] = df3['lat'].astype('float')
df3 = df3.rename(columns = {'lng':'lon'})

df_map = df3[['lat','lon']]

st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     initial_view_state=pdk.ViewState(
         latitude=42.36,
         longitude=-90.06,
         zoom=3,
     ),
     layers=[
         pdk.Layer(
             'ScatterplotLayer',
             data=df_map,
             get_position='[lon, lat]',
             get_color='[50, 158, 168, 70]',
             get_radius=15000,
         ),
     ],
 ))


## count the number of protests each day
st.subheader("The number of protests by date and day")
df1 = df.groupby('Date').size().reset_index().rename(columns={0:'Value'}).sort_values(by='Value',ascending=False)
df1['Date'] = pd.to_datetime(df1['Date'])
df1['Day_of_week'] = df1['Date'].dt.day_name()
# st.dataframe(df1)

#1
# st.bar_chart(df1)

#2
# plt.figure(figsize = (10,8))
# plt.bar(df1['Date'], df1['Value'],align='center', alpha=0.5)
# plt.xticks( rotation=-35)
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.legend('Date')
# plt.title('Number of protests each day')
# st.pyplot()

#3
days = df1["Day_of_week"].unique()
my_list = ["Monday",'Tuesday','Wednesday','Thursday','Friday','Saturday',"Sunday"]
day_list = st.selectbox("Select Day", my_list)
st.write("You selected:",day_list)

def get_rest(selection):
    li = my_list
    li.remove(selection)
    return li

chart_placeholder = st.empty()
c = alt.Chart(df1).mark_bar(color = "lightgray", size=15).encode(
    # x=alt.X('Date', axis=alt.Axis(tickCount=df1['Date'], title='Date')),
    x = 'Date',
    y ='Value:Q',
    color = alt.Color('Day_of_week',
            scale=alt.Scale(
                domain = [day_list] + get_rest(day_list),
                range=['lightseagreen','lightgray','lightgray','lightgray','lightgray','lightgray','lightgray'])
                ),
    tooltip = ['Date','Value']
).properties(
    width = 900,
    height = 600
)
chart_placeholder.altair_chart(c)


## show the number of protests across the country by state and date
st.subheader("The number of protests in each state by date")
st.write("*The size of the dot corresponds to the number of protests.*") # Italics

df2 = pd.read_excel("data/date_state_num.xlsx")
df2['date'] = df2['date'].astype(str)

pd.set_option('display.max_columns', None)

fig = px.scatter_geo(df2, locationmode='USA-states', locations='state_id',
    size='num_protests', hover_name='state_id',
    projection="albers usa",
    size_max=50, animation_frame='date')
fig.update_layout(width=900,height=600,showlegend = True)

st.plotly_chart(fig)


## Arrests distribution across states
st.subheader("Arrests distribution across states")

df4 = df
df4['State_name'] = df4['State_name'].astype('str')
df4 = df4.groupby('State_name')['ReportedArrests'].sum().reset_index()

chart_placeholder1 = st.empty()
c1 = alt.Chart(df4).mark_bar().encode(
    # x=alt.X('Date', axis=alt.Axis(tickCount=df1['Date'], title='Date')),
    x='State_name:O',
    y='ReportedArrests:Q',
    tooltip = ['State_name','ReportedArrests']
).properties(
    width = 900,
    height = 600
)
chart_placeholder1.altair_chart(c1)


## Injuries and Damages across the country
st.subheader("Injuries and Protesters distribution across states")

df5 = df
df5['State_name'] = df5['State_name'].astype('str')
df5 = df5.rename(columns = {'ReportedArrests':'Arrests','ReportedParticipantInjuries':'Participant_injuries',
'ReportedPoliceInjuries':'Police_injuries','ReportedPropertyDamage':'Property_damage'})
df5 = df5.groupby('State_name')['Participant_injuries','Police_injuries','Property_damage'].sum()
df5 = df5.stack().reset_index()
df5.columns = ['State_name','Category','Value']
df5['Category'] = df5['Category'].astype('str')

chart_placeholder2 = st.empty()
c2 = alt.Chart(df5).mark_bar().encode(
    x=alt.X('State_name:N',title=None),
    y=alt.Y('Value:Q',axis=alt.Axis(grid=False,title=None)),
    color=alt.Color('Category:N',scale=alt.Scale(range=['#96ceb4', '#ffcc5c','#ff6f69','blue'])),
    tooltip = ['State_name','Category','Value']
).properties(
    width = 1100,
    height = 700
)
chart_placeholder2.altair_chart(c2)
