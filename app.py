import streamlit as st
import numpy as np
import pickle
import pandas as pd


filename = "scaled_data.csv"
scaled_data = pd.read_csv(filename)

X = scaled_data.drop(columns = ['gdp','unemployment_rate', 'remittance_to_gdp_ratio']) # input features
Y = scaled_data[['gdp','unemployment_rate', 'remittance_to_gdp_ratio']]


# feature_name = ['population', 'remittance_growth_rate', 'remittance_paid', 'remittance_per_capita', 'remittance_received', 
#                 'remittance_volatility', 'unemployment_rate_change', 'net_migration', 'remittance_to_gdp_ratio','country_name',
#                 'new_income_group','new_region']

feature_name = ['Year', 'remittance_received', 'country_name', 'new_income_group', 'new_region']


path = "randomfmodel.pkl"

def load_model(path):
    with open(path, "rb") as mod:
        model = pickle.load(mod)
    return model

with open('1min_max_dict.json', 'rb') as fp:
    min_max_dict = pickle.load(fp)

with open('1cat_dict.json', "rb") as fp:
    cat_dict = pickle.load(fp)


def get_value(feature_name, value, my_dict = cat_dict):
    feature_dict = my_dict[feature_name]
    Encodedvalue = feature_dict[value]
    return Encodedvalue


def get_feature_dic(feature_name, dic = cat_dict):
    return dic[feature_name]

def scale_value(feature_name, value, dic = min_max_dict):
    maximum = dic[feature_name]['max']
    minimum = dic[feature_name]['min']
    scaled_value = (value - minimum)/ (maximum - minimum)
    return scaled_value





def main():
    # Bank Churn Prediction App
    st.title("Remmitance Prediction")
    activities = ["Introduction", "Predictions", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    st.sidebar.markdown(
            """ Developed by Team Catboost
                """)

    if choice == "Introduction":
        html_temp_home1 = """<div style="background-color:#00CCFF;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Remmitance prediction Using Machine Learning Model
                                            </h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
    
        st.write("""
            """)
        st.write("""
                 
                 
                 """)
        
        
    elif choice == "Predictions":
        st.subheader("Remmitance")
        
        Year_class = st.number_input("Enter the year:")
        Year_scaled = scale_value("Year", Year_class)

        remittance_received_class = st.number_input("Enter remittance Received:")
        remittance_received_scaled = scale_value("remittance_received", remittance_received_class)
              
        country_name_class = st.selectbox("Select Country", tuple(get_feature_dic("country_name").keys()))
        country_name_code = get_value(feature_name="country_name",value=country_name_class)
        country_name_scaled = scale_value("country_name", country_name_code)
        
        new_income_group_class = st.selectbox("Select Income Group", tuple(get_feature_dic("new_income_group").keys()))
        new_income_group_code = get_value(feature_name = "new_income_group", value = new_income_group_class)
        new_income_group_scaled = scale_value("new_income_group", new_income_group_code)
        
        new_region_class = st.selectbox("Select Region", tuple(get_feature_dic("new_region").keys()))
        new_region_code = get_value(feature_name = "new_region", value = new_region_class)
        new_region_scaled = scale_value("new_region", new_region_code)    
        
        
        feature_values = [Year_scaled, remittance_received_scaled, country_name_scaled, new_income_group_scaled, new_region_scaled]


        
    
        
        single_sample = np.array(feature_values).reshape(1,-1)
        single_sample = [feature_values]

        
        if st.button("Predict"):
            df = pd.DataFrame(single_sample , columns = feature_name)
            preds = load_model(path).predict(df)
            st.write(f"The GDP is: {preds[0][0]}")
            st.write(f"The remittance to GDP ratio is : {preds[0][1]:.4f}")
            st.write(f"The unemployment rate is: {preds[0][2]:.2f}")

                
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#4073FF;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Definition : </h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                                        <div style="background-color:#4073FF;padding:10px">
                                        <h4 style="color:white;text-align:center;">This application is developed by team Catboost for the purpose of .</h4>
                                        <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                                        </div>
                                        <br></br>
                                        <br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

if __name__ == "__main__":
    main()