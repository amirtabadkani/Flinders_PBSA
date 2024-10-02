import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


import numpy as np

st.set_page_config(page_title='Flinders University - Adelaide', layout='wide')

hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html= True)


CHEP_en = pd.read_csv('./data/Flinders.csv')
CHEP_ML = pd.read_csv('./data/Flinders_ML.csv')


# CHEP_en = pd.read_csv(r'C:\Users\atabadkani\StreamlitApps\FlindersEnCO2\data\Flinders.csv')

# CHEP_ML = pd.read_csv(r'C:\Users\atabadkani\StreamlitApps\FlindersEnCO2\data\Flinders_ML.csv')


st.title(":red[Flinders University PBSA] - Building Performance Analytical Dashboard")

st.markdown('---')

Floor_area = 1555 ##########
en_price = 0.45 ###$/kWh
REF_EUI = 18.58
REF_ELECp = 19.66
REF_DA = 27.64
REF_glare = 1.49
REF_Cost = REF_EUI*Floor_area*en_price

#Energy Cost Calc

CHEP_en['Energy Cost (AU$/yr)'] = CHEP_en['EUI (kWh/m2)']*Floor_area*en_price
CHEP_ML['Energy Cost (AU$/yr)'] = CHEP_ML['EUI (kWh/m2)']*Floor_area*en_price

with st.sidebar:
    st.image('https://www.ceros.com/wp-content/uploads/2019/04/Stantec_Logo.png',use_column_width='auto',output_format='PNG')
    st.markdown('**_This tool is only built for demonstration purpose of Flinders PBSA project based in Adelaide to better understand the impact of different design variables on building energy and indoor visual/thermal comfort performance. Results should be interpreted as comparative only, and do not aim to predict actual building performance._**')

    
    st.title('Select Your Desired Design Variables:')
    
    WWR_NS = st.select_slider('Window-to-Wall Ratio (North-South):', options = [25, 35, 45], value = 25, key = 'WWR_NS') 
    WWR_EW = st.select_slider('Window-to-Wall Ratio (East-West):', options = [25, 35, 45], value = 25, key = 'WWR_EW')
    #st.markdown("_WWR represents the amount of window area compared to the total wall area for the respective building facades. Higher ratios mean more window area, which can lead to more daylight but also potentially more heat gain or loss, depending on the climate and the window's properties._")

    Shade_type = st.select_slider('Shade Type:', options = ['No Shade','1m Overhang','Screen'], value = 'No Shade', key = 'shadetype')
    #st.markdown("_The depth of shading devices like overhangs or fins on a building's windows, measured in millimeters. Shading devices can significantly reduce the amount of solar heat gain in a building, improving comfort and reducing cooling loads._")
    
    SHGC = st.select_slider('Glass SHGC:', options = [0.25,0.35,0.45], value = 0.35, key = 'glass_shgc')
    if SHGC == 0.25:
        st.markdown("**VLT equal to 0.33**")
    elif SHGC == 0.35:
        st.markdown("**VLT equal to 0.43**")
    elif SHGC == 0.45:
        st.markdown("**VLT equal to 0.53**")

    Uvalue = st.select_slider('Glass U-value:', options = [2.5,3.5], value = 3.5, key = 'glass_u-value')

    exwall = st.select_slider('External Wall R-value:', options = [1.0,2.8], value = 2.8, key = 'exwall_r')
    #st.markdown("_The R-value measures the thermal resistance of a material or assembly like a wall. It's measured in m²/kw. The higher the R-value, the better the material or assembly is at resisting heat flow, which can reduce heating and cooling loads._")

    ventilation  = st.select_slider('Ventilation Type:', options = ['Natural','Mechanical'], value = 'Natural', key = 'venttype')
    
CHEP_en_RESULT = CHEP_en[CHEP_en['WWR-NS'].isin([WWR_NS]) & CHEP_en['WWR-EW'].isin([WWR_EW]) & CHEP_en['ShadeType'].isin([Shade_type]) & CHEP_en['SHGC'].isin([SHGC]) & CHEP_en['U-value'].isin([Uvalue]) & CHEP_en['ExWall'].isin([exwall]) & CHEP_en['Ventilation'].isin([ventilation])]

with st.container():
     
    st.subheader(f'Design Selection Summary - Iteration {CHEP_en_RESULT.index[0]} ')
  
    def get_metrics_EUI():
       
        CHEP_en_RESULT = CHEP_en[CHEP_en['WWR-NS'].isin([WWR_NS]) & CHEP_en['WWR-EW'].isin([WWR_EW]) & CHEP_en['ShadeType'].isin([Shade_type]) & CHEP_en['SHGC'].isin([SHGC]) & CHEP_en['U-value'].isin([Uvalue]) & CHEP_en['ExWall'].isin([exwall]) & CHEP_en['Ventilation'].isin([ventilation])]
        
        EUI_METRIC = CHEP_en_RESULT['EUI (kWh/m2)']
        ELEC_P = CHEP_en_RESULT['ELECp (W/m2)']
        AVE_DA =  CHEP_en_RESULT['Daylight Autonomy']
        AVE_UDI = CHEP_en_RESULT['Excessive Daylight']
        ENERGY = CHEP_en_RESULT['Energy Cost (AU$/yr)']
        Space_NE = CHEP_en_RESULT['OT26% - L5-Student-Com-NE']
        Space_SW = CHEP_en_RESULT['OT26% - L5-Studio-SW']
        Space_SE= CHEP_en_RESULT['OT26% - L5-Studio-SE']
        Space_NW = CHEP_en_RESULT['OT26% - L5-Studio-NW']
        image = CHEP_en_RESULT['img']
        EUI_REF = CHEP_en_RESULT['EUI Saved(-)Wasted(+)']
        ELEC_REF = CHEP_en_RESULT['REF_ELECp']
        DA_REF = CHEP_en_RESULT['REF_DA']
        UDI_REF = CHEP_en_RESULT['REF_ExcDA']
        # ventilation = CHEP_en_RESULT['Ventilation']
        
        return EUI_METRIC, ELEC_P, AVE_DA, AVE_UDI, ENERGY, Space_NE, Space_SW, Space_SE, Space_NW, image, EUI_REF,ELEC_REF,DA_REF,UDI_REF,ventilation
            
    get_metrics_EUI()

    cols = st.columns([0.1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.1])

    with cols[0]:
        ""
    with cols[1]:
        st.metric('EUI', f'{round((get_metrics_EUI()[0].iloc[0]),2)}kWh/m2')
    with cols[2]:
        st.metric('ELECp', f'{round((get_metrics_EUI()[1].iloc[0]),2)}W/m2')
    with cols[3]:
        st.metric('Daylight Autonomy (500lx)', f'{round((get_metrics_EUI()[2].iloc[0]),2)}%')
    with cols[4]:
        st.metric('Excessive Daylight (>10000lx)',f'{round((get_metrics_EUI()[3].iloc[0]),2)}%')
    with cols[5]:
        st.metric('Operational Cost', f'{format(int(get_metrics_EUI()[4]), ",d")} AU$/yr')
    with cols[6]:
        st.metric('Comm Block NE (> OT 26C)', f'{round((get_metrics_EUI()[5].iloc[0]),2)}%')
    with cols[7]:
        st.metric('Studio Block SW (> OT 26C)', f'{round((get_metrics_EUI()[6].iloc[0]),2)}%')
    with cols[8]:
        st.metric('Studio Block SE (> OT 26C)', f'{round((get_metrics_EUI()[7].iloc[0]),2)}%')
    with cols[9]:
        st.metric('Studio Block NW (> OT 26C)', f'{round((get_metrics_EUI()[8].iloc[0]),2)}%')
    with cols[10]:
        ""

    st.markdown(":red[**_Energy Use Intensity (EUI):_**] The Energy Use Intensity measures a building's energy use per unit area, usually expressed in kWh/m2 per year. It's a key metric for assessing a building's energy efficiency. Lower EUI values mean better energy efficiency.")
    st.markdown(":red[**_Peak Electricity Load (ELECp):_**] This is the maximum electricity demand per unit area of the building, measured in W/m2. High peak loads can lead to higher utility costs, especially in areas where utilities charge based on peak demand.")
    # st.markdown(":red[**_Cooling Thermal Peak (CLGp):_**] This is the maximum thermal cooling load per unit area of the building, measured in W/m2. High peak cooling loads can indicate potential issues with overheating and could require larger or more efficient cooling systems.")
    st.markdown(":red[**_Daylight Autonomy (%>500lux):_**] This is the percentage of a space that receives at least 500 lux from daylight alone for at least half of the operating hours each year. Daylight autonomy is a measure of a building's potential for daylighting, which can reduce the need for artificial lighting and improve occupant well-being.")
    st.markdown(":red[**_Excessive Daylight (%>10000lux):_**] This metric indicates the percentage of a space that receives more than 10,000 lux, which can be uncomfortably bright and lead to glare. High values can indicate potential issues with glare and may require adjustments to the window design or shading.")
    st.markdown(":red[**_Operative Temperature % Time > 26 deg C:_**] This metric measures the percentage of time the operative temperature in the respective zones exceeds 26°C. The operative temperature is a measure of thermal comfort that takes into account air temperature, radiant temperature, and air speed. High values could indicate potential issues with overheating.")

    
    st.subheader('Design Performance against Reference Case')
    
    data_ref_eui = {'WWR':'40%','Shade Type':'No Shades','U-Value/SHGC/VLT': '3.91 / 0.30 / 0.38', 'EXT Walls':'R1.4', 'INT Walls':'R1.4'}
    REF_DF_eui = pd.DataFrame([data_ref_eui], index = ['DtS Reference Case'])
    st.dataframe(REF_DF_eui, use_container_width= True)
          
    cols = st.columns(5)
            
    with cols[0]:
        eui_performance = go.Figure(go.Indicator(
        mode = "gauge+delta",
        value = get_metrics_EUI()[0].iloc[0],
        delta = {'reference':REF_EUI, 'relative': True,'increasing':{'color':'red'}, 'decreasing':{'color':'Green'}, 'valueformat':".2%"},
        gauge = {'bar':{'color':'white', 'line':{'width':3}},'bordercolor':'darkgray','bgcolor':'lightcyan', 'threshold':{'thickness':1,'value':REF_EUI,'line': {'color': "red", 'width': 4}}, 'axis':{'dtick':2,'range':[1,50]}},
        number = {'font':{'family':'Arial', 'size':1}},
        title = {'text': "EUI Performance",'font':{'family':'Arial'}}
        ))

        st.plotly_chart(eui_performance, use_container_width=True)
    
    with cols[1]:
        elec_performance = go.Figure(go.Indicator(
        mode = "gauge+delta",
        value = get_metrics_EUI()[1].iloc[0],
        delta = {'reference':REF_ELECp,  'relative': True,'increasing':{'color':'red'}, 'decreasing':{'color':'Green'},'valueformat':".2%"},
        gauge = {'bar':{'color':'white', 'line':{'width':3}},'bordercolor':'darkgray','bgcolor':'lightcyan', 'threshold':{'thickness':1,'value':REF_ELECp,'line': {'color': "red", 'width': 4}}, 'axis':{'dtick':1,'range':(CHEP_en['ELECp (W/m2)'].min(),CHEP_en['ELECp (W/m2)'].max())}},
        number = {'font':{'family':'Arial'}},
        title = {'text': "ELECp Performance",'font':{'family':'Arial'}}
        ))
    
        st.plotly_chart(elec_performance, use_container_width=True)
    
    with cols[2]:
        elec_performance = go.Figure(go.Indicator(
        mode = "gauge+delta",
        value = get_metrics_EUI()[4].iloc[0]/1000,
        delta = {'reference':REF_Cost/1000,  'relative': True,'increasing':{'color':'red'}, 'decreasing':{'color':'Green'},'valueformat':"0.0%"},
        gauge = {'bar':{'color':'white', 'line':{'width':3}},'bordercolor':'darkgray','bgcolor':'lightcyan', 'threshold':{'thickness':1,'value':REF_Cost/1000,'line': {'color': "red", 'width': 4}}, 'axis':{'dtick':1,'range':[1,20]}},
        number = {'font':{'family':'Arial'}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Operational Energy Cost (1000$)",'font':{'family':'Arial'}}
        ))
    
        st.plotly_chart(elec_performance, use_container_width=True)

    with cols[3]:
        DA_performance = go.Figure(go.Indicator(
        mode = "gauge+delta",
        value = get_metrics_EUI()[2].iloc[0],
        delta = {'reference':REF_DA, 'relative': True,'increasing':{'color':'Green'}, 'decreasing':{'color':'Red'},'valueformat':".2%"},
        gauge = {'bar':{'color':'white', 'line':{'width':3}},'bordercolor':'darkgray','bgcolor':'lightcyan', 'threshold':{'thickness':1,'value':REF_DA,'line': {'color': "red", 'width': 4}}, 'axis':{'dtick':5,'range':[0,100]}},
        number = {'font':{'family':'Arial'}},
        title = {'text': "Daylight Level",'font':{'family':'Arial'}}
        ))
    
        st.plotly_chart(DA_performance, use_container_width=True)
    
    with cols[4]:
        glare_performance = go.Figure(go.Indicator(
        mode = "gauge+delta",
        value = get_metrics_EUI()[3].iloc[0],
        delta = {'reference':REF_glare,'relative': True,'increasing':{'color':'red'}, 'decreasing':{'color':'Green'},'valueformat':".2%"},
        gauge = {'bar':{'color':'white', 'line':{'width':3}},'bordercolor':'darkgray','bgcolor':'lightcyan', 'threshold':{'thickness':1,'value':REF_glare,'line': {'color': "red", 'width': 4}}, 'axis':{'dtick':1,'range':[0,15]}},
        number = {'font':{'family':'Arial'}},
        title = {'text': "Discomfort Glare Level",'font':{'family':'Arial'}}
        ))
    
        st.plotly_chart(glare_performance, use_container_width=True)
  
        
    def loadImages():
      
        # img = Image.open(rf'C:\Users\atabadkani\StreamlitApps\FlindersEnCO2\data\images\{get_metrics_EUI()[9].iloc[0]}')
        img = Image.open(f'./data/images/{get_metrics_EUI()[9].iloc[0]}')
        # ref = Image.open(r'C:\Users\atabadkani\StreamlitApps\FlindersEnCO2\data\images\REF02.png')
        ref = Image.open('./data/images/REF02.png')
        return img,ref
    
    col1,col2,col3,col4 = st.columns([0.5,4,0.5,4])
    
    with col1:
        ""
    with col2:
        st.subheader(':Gray[**Proposed Design Iteration**]')
        st.image(loadImages()[0], caption="", use_column_width = True)
    with col3:
        ""
    with col4:
        st.subheader(':Gray[**DtS Reference Design**]')
        st.image(loadImages()[1], caption='', use_column_width = True)
    
    st.markdown("---")

    st.header("**Design Inputs vs. Building Performance**")
    
    # columns = ['WWR-NS', 'WWR-EW', 'ShadeType', 
    #    'SHGC', 'ExWall', 'Ventilation','EUI (kWh/m2)', 'ELECp (W/m2)', 'EUI Saved(-)Wasted(+)',
    #    'Daylight Autonomy', 'Excessive Daylight', 'Energy Cost (AU$/yr)',
    #    'OT26% - L5-Student-Com-NE', 'OT26% - L5-Studio-SW','OT26% - L5-Studio-SE', 'OT26% - L5-Studio-NW']

    # # Determine the max EUI value
    # max_eui = CHEP_ML['EUI (kWh/m2)'].max()
    # min_eui = CHEP_ML['EUI (kWh/m2)'].min()
    # cols = st.columns([0.1,2,0.1])
    # with cols[0]:
    #     ""
    # with cols[1]:

    #     chep_pcm = px.parallel_coordinates(CHEP_ML, columns, color="EUI (kWh/m2)",
    #                                     color_continuous_scale=px.colors.cyclical.Edge,
    #                                     range_color=[min_eui, max_eui],
    #                                     color_continuous_midpoint=7, height = 800, width = 2500)
        
    #     chep_pcm.update_layout(coloraxis=dict(cmin=min_eui), coloraxis_showscale=True,font=dict(size=17),margin_autoexpand= True,autosize=True,xaxis=dict(domain=[0, 1]))
        
    #     st.plotly_chart(chep_pcm, use_container_width=True)

    #     st.markdown("**Shade Type: {[0] = No Shades, [1] = 1m Overhang, and [2] = External Mesh Screen with 50% Opacity}**")
    #     st.markdown("**Ventilation Type: {[0] = Mechanical Ventilation + Heating/Cooling, and [1] = Natural Ventilation + Heating/Cooling}**")
    
    # with cols[2]:
    #     ""


with st.container():
    
    st.components.v1.iframe("https://tt-acm.github.io/DesignExplorer/?ID=BL_47PJb2m", height = 1300)

st.markdown("---")

with st.expander('Statistical Building Performance Correlations (Box Plots)'):

    chep_bx_01 = px.box(CHEP_en, CHEP_en["WWR-NS"], CHEP_en["EUI Saved(-)Wasted(+)"], "WWR-NS", notched = True)
    chep_bx_02 = px.box(CHEP_en, CHEP_en["WWR-EW"], CHEP_en["EUI Saved(-)Wasted(+)"], "WWR-EW", notched = True)
    chep_bx_03 = px.box(CHEP_en, CHEP_en["ShadeType"], CHEP_en["EUI Saved(-)Wasted(+)"], "ShadeType", notched = True)
    chep_bx_04 = px.box(CHEP_en, CHEP_en["SHGC"], CHEP_en["EUI Saved(-)Wasted(+)"], "SHGC", notched = True)
    chep_bx_05 = px.box(CHEP_en, CHEP_en["ExWall"], CHEP_en["EUI Saved(-)Wasted(+)"], "ExWall", notched = True)
    chep_bx_06 = px.box(CHEP_en, CHEP_en["Ventilation"], CHEP_en["EUI Saved(-)Wasted(+)"], "Ventilation", notched = True)

    cols = st.columns(6)
    
    with cols[0]:
        st.plotly_chart(chep_bx_01, use_container_width=True)
    with cols[1]:
        st.plotly_chart(chep_bx_02, use_container_width=True)
    with cols[2]:
        st.plotly_chart(chep_bx_03, use_container_width=True)
    with cols[3]:
        st.plotly_chart(chep_bx_04, use_container_width=True)
    with cols[4]:
        st.plotly_chart(chep_bx_05, use_container_width=True)
    with cols[5]:
        st.plotly_chart(chep_bx_06, use_container_width=True)

    chep_bx_07 = px.box(CHEP_en, CHEP_en["WWR-NS"], CHEP_en["Daylight Autonomy"], "WWR-NS", notched = True)
    chep_bx_08 = px.box(CHEP_en, CHEP_en["WWR-EW"], CHEP_en["Daylight Autonomy"], "WWR-EW", notched = True)
    chep_bx_09 = px.box(CHEP_en, CHEP_en["ShadeType"], CHEP_en['Daylight Autonomy'], "ShadeType",  notched = True)
    chep_bx_10 = px.box(CHEP_en, CHEP_en["VLT"], CHEP_en['Daylight Autonomy'], "VLT",  notched = True)
    
    cols = st.columns(4)
    
    with cols[0]:
        st.plotly_chart(chep_bx_07, use_container_width=True)
    with cols[1]:
        st.plotly_chart(chep_bx_08, use_container_width=True)
    with cols[2]:
        st.plotly_chart(chep_bx_09, use_container_width=True)
    with cols[3]:
        st.plotly_chart(chep_bx_10, use_container_width=True)

    chep_bx_13 = px.box(CHEP_en, CHEP_en["WWR-NS"], CHEP_en["Excessive Daylight"], "WWR-NS", notched = True)
    chep_bx_14 = px.box(CHEP_en, CHEP_en["WWR-EW"], CHEP_en["Excessive Daylight"], "WWR-EW", notched = True)
    chep_bx_15 = px.box(CHEP_en, CHEP_en["ShadeType"], CHEP_en['Excessive Daylight'], "ShadeType",  notched = True)
    chep_bx_16 = px.box(CHEP_en, CHEP_en["VLT"], CHEP_en['Excessive Daylight'], "VLT",  notched = True)
    
    cols = st.columns(4)
    
    with cols[0]:
        st.plotly_chart(chep_bx_13, use_container_width=True)
    with cols[1]:
        st.plotly_chart(chep_bx_14, use_container_width=True)
    with cols[2]:
        st.plotly_chart(chep_bx_15, use_container_width=True)
    with cols[3]:
        st.plotly_chart(chep_bx_16, use_container_width=True)

def get_index(df) -> dict:
        dict_ = {df['Version: EPiC Database 2019'].iloc[i]: i for i in range(0, len(df['Version: EPiC Database 2019']))}
        return dict_

#Ridge Regression
#-------------------------------------------------------------------------------------------------------------------------------------

CHEP_ridge = CHEP_ML.drop(['img', 'REF_ELECp','REF_DA','REF_ExcDA'], axis=1)

with st.expander('Statistical Building Performance Correlations (Heat Map)'):

  CHEP_CORR_HTM = px.imshow(round(CHEP_ridge.corr(),2),text_auto=True,color_continuous_scale='thermal',  width = 1000, height = 1000)
  CHEP_CORR_HTM.update_traces(textfont_size=15)
  
  st.plotly_chart(CHEP_CORR_HTM, use_container_width=True)
  
  st.markdown('**:red[Note:]** Numbers represent the magnitude level of variables against each other, and Negative Values mean the input impacts the target negatively.')

#Energy Tagets 

st.header("**Predicting Energy Usage Thorugh Artificial Intelligence (AI)**")

feature_energy_cols =  CHEP_ridge[['WWR-NS', 'WWR-EW', 'ShadeType', 'SHGC','U-value', 'ExWall', 'Ventilation']].columns

X = pd.DataFrame(CHEP_ridge[['WWR-NS', 'WWR-EW', 'ShadeType', 'SHGC', 'U-value', 'ExWall','Ventilation']], columns = feature_energy_cols)

y = CHEP_ridge[['EUI (kWh/m2)', 'ELECp (W/m2)', 'OT26% - L5-Student-Com-NE', 'OT26% - L5-Studio-SW','OT26% - L5-Studio-SE', 'OT26% - L5-Studio-NW','EUI Saved(-)Wasted(+)','Energy Cost (AU$/yr)']]

X_train_en, X_test_en, y_train_en, y_test_en = train_test_split(X,y,test_size=1)

ridge_en = Ridge(alpha=1.0)

ridge_en.fit(X_train_en,y_train_en)

CHEP_en_cdf = pd.DataFrame(np.transpose(ridge_en.coef_),X.columns,columns=[['EUI (kWh/m2)', 'ELECp (W/m2)', 'OT26% - L5-Student-Com-NE', 'OT26% - L5-Studio-SW','OT26% - L5-Studio-SE', 'OT26% - L5-Studio-NW','EUI Saved(-)Wasted(+)','Energy Cost (AU$/yr)']])
             

#Predictions

cols = st.columns([0.1,1,0.2,2,0.1])
with cols[0]:
    ""
with cols[1]:
    
    new_WWR_NS = st.number_input('Desired Window-to-Wall Ratio (North-South):', min_value = 0, max_value = 100, value = 30, key = 'new_WWR_NS')
    new_WWR_EW = st.number_input('Desired Window-to-Wall Ratio (East-West):', min_value = 0, max_value = 100, value = 50, key = 'new_WWR_EW')

    new_shade = st.number_input('Desired Shade Type:',min_value = 0, max_value = 2, value = 2, key = 'new_shadetype')
    if new_shade == 0:
        st.markdown("**No Shades**")
    elif new_shade == 1:
        st.markdown("**1m Overhang**")
    elif new_shade == 2:
        st.markdown("**External mesh screen with 50% opacity**")
    new_uvalue = st.number_input('Desired Glass U-value:', min_value = 0.5, max_value = 5.8, value = 1.8, key = 'new_uvalue')
    new_SHGC = st.number_input('Desired Glass SHGC:', min_value = 0.05, max_value = 0.9, value = 0.8, key = 'new_shgc')
    new_exwall = st.number_input('Desired External Wall R-value:',  min_value = 0.5, max_value = 6.0, value = 3.5 , key = 'new_rvalue')

    new_ventilation = st.number_input('Desired Ventilation Type:',  min_value = 0, max_value = 1, value = 1 , key = 'new_vent')
    if new_ventilation == 0:
        st.markdown("**Mechnaical Ventilation + Heating and Cooling**")
    elif new_ventilation == 1:
        st.markdown("**Natural Ventilation + Heating and Cooling**")


    new_vals_energy = np.array([new_WWR_NS,new_WWR_EW,new_shade,new_SHGC,new_uvalue,new_exwall,new_ventilation]).reshape(1,7)
    new_energy_df = pd.DataFrame(new_vals_energy, columns = feature_energy_cols)
    pred_energy = ridge_en.predict(new_energy_df)
    
    pred_energy = np.transpose(pred_energy.tolist())
    
with cols[2]:
    ""
with cols[3]:
    
    
    stackData = {"Predicted EUI":[int(pred_energy[0])],
                  "Reference EUI":[REF_EUI],
                  "Predicted ELECp":[int(pred_energy[1])],
                  "Reference ELECp":[REF_ELECp],
                  "Predicted Cost":[int(pred_energy[7]/1000)],
                  "Reference Cost":[int((REF_EUI*Floor_area*en_price)/1000)],
                  "EUI":["EUI (kWh/m2)"],
                  "ELECp":["ELECp (W/m2)"],
                  "Cost":["Operational Cost (1000$/yr)"]
                  }
    

    predicted_bar = go.Figure(data=[
                        go.Bar(name = "Predicted EUI", x = stackData["EUI"], y=stackData["Predicted EUI"],
                               offsetgroup=-1,marker_color = '#024a70', offset = -0.17, text = stackData["Predicted EUI"],textposition ='inside', width = 0.35, opacity = 0.8),
                        go.Bar(name = "Reference EUI", x = stackData["EUI"], y=stackData["Reference EUI"],
                               offsetgroup=-1,marker_color = '#051c2c', offset = -0.17,text = stackData["Reference EUI"], textposition ='inside', base=stackData["Predicted EUI"], width = 0.35, opacity = 0.8),
                        go.Bar(name = "Predicted ELECp", x = stackData["ELECp"],y=stackData["Predicted ELECp"],
                               offsetgroup=2,marker_color = '#abe5f0',offset = -0.17, text = stackData["Predicted ELECp"], textposition ='inside',  width = 0.35, opacity = 0.8),
                        go.Bar(name = "Reference ELECp", x = stackData["ELECp"],y=stackData["Reference ELECp"],
                               offsetgroup=2,marker_color = '#74d0f0',offset = -0.17, text = stackData["Reference ELECp"],textposition ='inside',  base=stackData["Predicted ELECp"], width = 0.35, opacity = 0.8),
                        go.Bar(name = 'Predicted Cost', x = stackData["Cost"],y=stackData['Predicted Cost'],
                               offsetgroup=4,marker_color = '#d62728',offset = -0.17,text = stackData["Predicted Cost"],textposition ='inside',width = 0.35, opacity = 0.8),
                        go.Bar(name = 'Reference Cost', x = stackData["Cost"],y=stackData['Reference Cost'],
                               offsetgroup=4,marker_color = '#ff7f0e',offset = -0.17,text = stackData["Reference Cost"],textposition ='inside', base=stackData["Predicted Cost"],width = 0.35, opacity = 0.8)],
        layout=go.Layout(
        yaxis_title="Predicted vs. Reference Building Performance"
    ))

    predicted_bar.update_layout(height = 500, width = 1000)
    
    st.plotly_chart(predicted_bar,use_container_width=True,use_container_height=True)
            
with cols[4]:
    ""


