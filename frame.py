# cd my file path
# $ streamlit run frame.py
# save and refresh the webpage whenever there's any change in codes

import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import altair as alt
import datetime as dt
import plotly.express as px
import pydeck as pdk
from PIL import Image


# set webpage width to 1100px
def _max_width_():
    max_width_str = f"max-width: 1100px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

_max_width_()


# Main title
# use st.header / st.subheader for section titles
st.title("U.S. Protests for Racial Justice")
st.write("Kyle Skene, Houjiang Liu, Yuan Hua")

st.markdown("---")
st.markdown(
"""
This project explores the facts about the protests against police 
violence sparked by George Floyd’s death from a data-driven perspective. 
It aims to uncover the influence and intensity of a series of protests and find hidden patterns of the
protest data temporally and spatially.
Based on the real data, we can interpret the protests from quantity, participants, 
location, violence involved, and changes over time. Through comprehensive 
data analysis, we want to use scientific methods to bring up the discourse of 
**racial equity and justice**.
"""
)


## clean the dataset
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

## Show our data
if st.checkbox('Show clean dataset'):
    st.dataframe(df)
st.markdown("---")


st.markdown("<h2 style = 'margin-bottom:0'><b>Data Visualization:</b></h2>", unsafe_allow_html=True)
##1 show each protest on the map (4017 rows of data)
st.subheader("Protests Distribution: May25 – Jun 28")
st.markdown(
"""
Protests have spread across the country, from big cities to small towns. The largest concentration of 
demonstrations is on the east and west coasts.
"""
)
st.write("*Each dot represents one protest in a city/town.*") # Italics
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


##2 show the number of protests across the country by state and date
st.subheader("The number of protests in each state by date")
st.markdown(
"""
Peaks of protests occurred mainly on weekends. This makes sense because most Americans have more free time on the weekends and do not have to work during this time. 
After the data’s maximum on June 6th, we noticed that the protests seem to be dwindling down. 
However, a resurgence of protests occurred on June 19th, otherwise known as Juneteenth. 
"""
)
st.write("*The size of the circle is proportional to the relative number of protests that occurred in each state.*") # Italics

df2 = pd.read_excel("data/date_state_num.xlsx")
df2['date'] = df2['date'].astype(str)

pd.set_option('display.max_columns', None)

fig = px.scatter_geo(df2, locationmode='USA-states', locations='state_id',
    size='num_protests', hover_name='state_id',
    projection="albers usa",
    size_max=50, animation_frame='date')
fig.update_layout(width=1200,height=600,showlegend = True)

st.plotly_chart(fig)

# the overall protests in each state
df6 = df.groupby('State_name').size().reset_index().rename(columns={0:'Value'}).sort_values(by = 'Value',ascending = False)
df6.reset_index(drop = True)
if st.checkbox('Show data of total protests in each state'):
    st.dataframe(df6, width = 600)


##3 count the number of protests each day
st.subheader("The number of protests by date and day")
st.markdown(
"""
Peaks of protests occurred mainly on weekends. This makes sense because most Americans have more free time on the weekends and do not have to work during this time. 
After the data’s maximum on June 6th, we noticed that the protests seem to be dwindling down. 
However, a resurgence of protests occurred on June 19th, otherwise known as Juneteenth. 
"""
)

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
    width = 1200,
    height = 600
)
chart_placeholder.altair_chart(c)


##4 
## Arrests distribution across states
st.subheader("Arrests distribution across states")
st.markdown(
"""
The overwhelming majority of states with protests had zero, or close to zero, reported arrests.
However, there are states like California, Nevada, Georgia, and Texas reported over 400 arrests from the protests between May and June. 
"""
)

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
    width = 1200,
    height = 600
)
chart_placeholder1.altair_chart(c1)


##5
## Injuries and Damages across the country
st.subheader("Injuries and Property damages across states")
st.markdown(
"""
Property damage was higher in the state where George Floyd was killed. 
"""
)

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
    width = 1200,
    height = 600
)
chart_placeholder2.altair_chart(c2)


st.markdown('---')
##6
## Data Modeling about protesters

# title
st.markdown("<h2 style = 'margin-bottom:0'><b>Data Modeling:</b></h2>", unsafe_allow_html=True)
st.markdown("<h3 style ='line-height:1.4; margin-top:0'>How possible might we use the demographic features to classify protest intensity?</h3>", unsafe_allow_html=True)
st.markdown(
"""
To use demographic features to classify protest intensity, we mainly explored how KNN and logistic 
regression perform in this classification problem.
"""
)

st.markdown("<h2 style = 'margin-bottom:0'>Preparing Data</h2>", unsafe_allow_html=True)
st.markdown(
"""
When we prepare the data, first, we merged the city protest data with demographic data. 
Second, we used interquartile range to remove the outliners. Third, we labeled the intensity 
of the city protest by setting a level threshold of the protest rate in the 50th percentiles 
distribution. 
"""
)

# show data
## Show our data
df7 = pd.read_csv("data/new_protest_data.csv")
df7 = df7.drop(columns=['Unnamed: 0'])
if st.checkbox('Show data description'):
    st.dataframe(df7.describe().T)

# formulas
st.markdown("")
# image1 = Image.open('img/protest_rate.png')
# st.image(image1,caption='Protest rate formula',width = 600)
# image2 = Image.open('img/outliers.png')
# st.image(image2,caption='Remove outliers formula')

imgs3 = ['img/protest_rate.png','img/outliers.png']
st.image(imgs3, width = 500)

# distribution
st.markdown("")
st.subheader("Clean Data Distribution")
st.markdown(
"""
From the box plot and the histogram of the cleaned dataset, we found that the protest
 rate is distributed normally and clustered around 0.006~0.007. 
"""
)
image3 = Image.open('img/clean_data_distribution.png')
st.image(image3, width = 1000)

# pair_plot and heatmap
st.subheader("Pair Plot and Heatmap")
st.markdown(
"""
From the plot below, we can find that the poverty level and the income show a high negative 
correlation. As they do not share the same correlation with the intensity, we keep both features 
as predictors in the classification modeling.
"""
)
# image4 = Image.open('img/pair_plot.png')
# st.image(image4,use_column_width=True)
# image5 = Image.open('img/heatmap_plot.png')
# st.image(image5,use_column_width=True)

imgs2 = ['img/pair_plot.png','img/heatmap_plot.png']
st.image(imgs2, width = 520)

st.markdown("")
st.markdown("")
st.markdown("<h2 style = 'margin-bottom:0'>Modeling Data</h2>", unsafe_allow_html=True)
st.markdown(
"""
When modeling the data, we first used KNN. To select the best K for KNN, we used a normal 10-fold 
cross-validation to calculate the accuracy. From the line plot (Figure 3) of different K values with 
their cross-validated scores, we find that the highest accuracy score of cross validation for K is 12.
"""
)

# st.subheader("k values plot")
# image6 = Image.open('img/k_values_plot.png')
# st.image(image6)

# st.subheader("roc curve KNN")
# image7 = Image.open('img/roc curve KNN.png')
# st.image(image7)

st.subheader("KNN Model")
imgs3 = ['img/k_values_plot.png','img/roc curve KNN.png']
st.image(imgs3, width = 500)

# table
st.write("*Confusion Matrix of the KNN result*")
df_knn_matrix = pd.DataFrame(
    np.array([[100,32],[60,71]]),
    columns=["Predicted:No","Predicted:Yes"]
)
df_knn_matrix.index = ["Actual: No", "Actual: Yes"]
st.dataframe(df_knn_matrix)

st.markdown("")
st.markdown(
"""
From the confusion matrix, we calculated the sensitivity of matrix by dividing the true positive by 
the sum of true positive and false negative. We found that the true positive rate is 0.54. This means 
that if there are 100 actual protests that are intense, only 54 of them will be correctly predicted. 
This means, the model is more likely to predict an actual intense protest as mild. This might mislead 
the local government to underestimate the protest intensity, thereby implement incomplete plans.
"""
)

st.subheader("Logistic Regression Model")
image8 = Image.open('img/roc curve LG.png')
st.image(image8)

# table2
st.write("*Confusion Matrix of the LG result*")
df_lg_matrix = pd.DataFrame(
    np.array([[78,54],[39,92]]),
    columns=["Predicted:No","Predicted:Yes"]
)
df_lg_matrix.index = ["Actual: No", "Actual: Yes"]
st.dataframe(df_lg_matrix)

st.markdown(
"""
As the results from the KNN model are not good enough for practical use in the real risk planning, 
we used the logistic regression (LG) to model the data. When evaluating results from both models, 
we first compare the cross-validated accuracy, the LG is 0.618, which is less than the accuracy of 0.626 
from the KNN model. However, the sensitivity of the LG is 0.702 which shows a better result than the KNN 
of which the sensitivity is 0.542.
"""
)


st.header('Model Comparison')
st.markdown(
"""
Finally, we find tuning the parameters of the LG model with its C and the penalty. 
We used the pipe function from the scikit-learn library and combined the C and penalty options. 
Then, we used grid search to generate a grid of parameter values and compared those values with 
AUC score.
"""
)

# table3
st.write("*Comparison between KNN and LG*")
df_comparison_models = pd.DataFrame(
    np.array([[0.626,0.618],[0.649,0.671],[0.542,0.702]]),
    columns=["KNN","Logistic Regression"]
)
df_comparison_models.index = ["Accuracy", "ROC_AUC", "True Positive Rate"]
st.dataframe(df_comparison_models)

# table4
st.write("*Performance between raw LG model and fine-tuning LG model*")
df_comparison_lg = pd.DataFrame(
    np.array([[0.671],[0.678]]),
    columns=["ROC_AUC"]
)
df_comparison_lg.index = ["raw model", "C=0.1, penalty=l1"]
st.dataframe(df_comparison_lg)
