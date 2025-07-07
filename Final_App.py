import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import sqrt
import random


def get_sample_input():
    return {
        'Rod_Piston_Engage_Dia(d_r)': 39.2,
        'Rod_Piston_Engage_Length(e_r)': 64,
        'Rod_Eye_Thickness(OD-ID/2)(t_re)': 20,
        'Pin_width(m_re)': 72,
        'Rod_Eye_Least_Thickness(t_l)': 15,
        'Thickness(o-i/2)(t_t)': 8,
        'Length(l_t)': 466,
        'Port_hole_dia_RHS(h_r)': 9,
        'Port_SpotFace_dia_RHS(d_r)': 13,
        'Port_spotface_width_RHS(w_r)': 1.5,
        'CEC-Tube_weld_Strength(Kg/sqmm)': 60,
        'CEC-Tube_weld_Radius(R)': 3,
        'CEC-Tube_weld_angle(a)': 4,
        'CEC-Tube_weld_depth(d)': 7,
        'Where(CEC_or_Tube)': 'CEC',
        'port_hole_dia_near_CEC(h_l)': 7,
        'Port_Spot_Face_dia_near_CEC(d_l)': 13,
        'HEC_Inner_dia(i_h)': 50,
        'Engage_Length(e_h)': 59,
        'CEC_Thickness(OD-ID)(t_c)': 16.5,
        'Pin_width(m_c)': 109,
        'Counterbore_ratio(r_c)': 1.147,
        'Piston_Thickness(o-i/2)(t_p)': 26.5,
        'Piston_Length(l_p)': 75,
        'Cushioning': 'NC',
        'Working Pressure': 230,
        'STROKE': 325,
        'BORE': 95,
        'ROD DIA': 45,
        'Test_Pressure': 400,
        'Target': 250000
    }


# Streamlit UI Setup
st.set_page_config(page_title="Final_App", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prediction on Historical Data</h1>", unsafe_allow_html=True)

# Load trained neural network model and preprocessing bundle
nn_bundle = joblib.load("D:/Lab/Model/nn_2025-07-04_15-18-13.pkl")
preprocess_bundle = joblib.load("./preprocessing_bundle.pkl")

# Extract model and preprocessing elements
nn_model = nn_bundle["model"]
scaler = preprocess_bundle["scaler"]
ohe = preprocess_bundle["ohe"]
cat_cols = preprocess_bundle["cat_cols"]
num_cols = preprocess_bundle["num_cols"]
nn_features = preprocess_bundle["feature_names"]

# Placeholder for prefill values from training data
# If data is removed for public hosting, initialize with sample defaults
# def get_sample_input():
#     sample = {col: 1.0 for col in num_cols}
#     sample.update({col: ohe.categories_[i][0] for i, col in enumerate(cat_cols)})
#     return sample

if 'prefill_row' not in st.session_state:
    st.session_state.prefill_row = get_sample_input()
prefill_row = st.session_state.prefill_row

# Define UI input layout groups
groupings = {
    "Basic Dimensions": ['STROKE', 'BORE', 'ROD DIA', 'Cushioning', 'Working Pressure'],
    "Rod": ['Rod_Piston_Engage_Dia(d_r)', 'Rod_Piston_Engage_Length(e_r)'],
    "Rod Eye": ['Rod_Eye_Thickness(OD-ID/2)(t_re)', 'Pin_Dia(dp_re)', 'Pin_width(m_re)', 'Rod_Eye_Least_Thickness(t_l)'],
    "Piston": ['Piston_Thickness(o-i/2)(t_p)', 'Piston_Length(l_p)'],
    "Tube": ['Thickness(o-i/2)(t_t)', 'Length(l_t)'],
    "CEC": ['CEC_Thickness(OD-ID)(t_c)', 'Pin_Dia(dp)', 'Pin_width(m_c)', 'Counterbore_ratio(r_c)'],
    "HEC": ['HEC_Inner_dia(i_h)', 'Engage_Length(e_h)'],
    "CEC-Tube weld": ['CEC-Tube_weld_Strength(Kg/sqmm)', 'CEC-Tube_weld_Radius(R)', 'CEC-Tube_weld_angle(a)', 'CEC-Tube_weld_depth(d)'],
    "HEC Port": ['Port_hole_dia_RHS(h_r)', 'Port_SpotFace_dia_RHS(d_r)', 'Port_spotface_width_RHS(w_r)'],
    "CEC port": ['Where(CEC_or_Tube)', 'port_hole_dia_near_CEC(h_l)', 'Port_Spot_Face_dia_near_CEC(d_l)'],
    "Test Parameters": ['Test_Pressure', 'Target']
}

# Collect Input
data_input = {}
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Enter Cylinder Parameters")
    for section, fields in groupings.items():
        with st.expander(section, expanded=True):
            for field in fields:
                if field in num_cols:
                    data_input[field] = st.number_input(field, value=float(prefill_row.get(field, 1.0)))
                elif field in cat_cols:
                    options = list(ohe.categories_[cat_cols.index(field)])
                    data_input[field] = st.selectbox(field, options, index=0)
                else:
                    key = f'input_{field}'
                    data_input[field] = st.number_input(label=field, key=key)
                

# Predict & FOS on button click
with right_col:
    st.subheader("Prediction")
    if st.button("Predict Result"):
        input_df = pd.DataFrame([data_input])

        # Preprocessing
        input_cat = ohe.transform(input_df[cat_cols])
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        cat_df = pd.DataFrame(input_cat, columns=cat_feature_names)
        if cat_feature_names[0] in cat_df.columns:
            cat_df.drop(columns=[cat_feature_names[0], cat_feature_names[-1]], inplace=True)
        num_scaled = scaler.transform(input_df[num_cols])
        num_df = pd.DataFrame(num_scaled, columns=num_cols)
        X_final = pd.concat([num_df, cat_df], axis=1)
        for col in nn_features:
            if col not in X_final:
                X_final[col] = 0
        X_final = X_final[nn_features]

        # Prediction
        result = nn_model.predict(X_final)[0]
        label = "Failed" if result == 1 else "Passed"
        color = "red" if result == 1 else "green"

        st.markdown(f"""
        <div style="padding: 30px; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
            <h2>{label}</h2>
        </div>
        """, unsafe_allow_html=True)







        # ---------------- FOS Calculations ------------------
        st.subheader("Factor of Safety (FOS) Summary")

        # Extract input features used in FOS calculations
        BORE = data_input['BORE']
        ROD_DIA = data_input['ROD DIA']
        STROKE = data_input['STROKE']
        Working_Pressure = data_input['Working Pressure']
        Test_Pressure = data_input['Test_Pressure']
        Thickness = data_input['Thickness(o-i/2)(t_t)']
        Pin_width_RE = data_input['Pin_width(m_re)']
        Rod_Piston_Engage_Dia = data_input['Rod_Piston_Engage_Dia(d_r)']
        Rod_Piston_Engage_Length = data_input['Rod_Piston_Engage_Length(e_r)']
        Rod_Eye_Thickness = data_input['Rod_Eye_Thickness(OD-ID/2)(t_re)']
        Pin_dia_RE = data_input['Pin_Dia(dp_re)']
        Rod_Eye_Least_Thickness = data_input['Rod_Eye_Least_Thickness(t_l)']
        Piston_Length = data_input['Piston_Length(l_p)']
        Piston_Thickness = data_input['Piston_Thickness(o-i/2)(t_p)']
        Pin_dia_CEC = data_input['Pin_Dia(dp)']
        CEC_Thickness = data_input['CEC_Thickness(OD-ID)(t_c)']
        Pin_width_CEC = data_input['Pin_width(m_c)']
        HEC_Inner_dia = data_input['HEC_Inner_dia(i_h)']
        Engage_Length = data_input['Engage_Length(e_h)']
        CEC_Tube_weld_Strength = data_input['CEC-Tube_weld_Strength(Kg/sqmm)']
        Port_hole_dia_RHS = data_input['Port_hole_dia_RHS(h_r)']
        port_hole_dia_near_CEC = data_input['port_hole_dia_near_CEC(h_l)']


        # Constants
        E = 2100000
        F = (np.pi/4) * (BORE)**2 * Working_Pressure
        Fex = (np.pi/4) * ((BORE)**2 - (ROD_DIA)**2) * Working_Pressure

        # Rod
        Rod_Tensile = 3600
        Rod_Shear = 3780
        Rod_Axial_FOS = (Rod_Tensile * np.pi * (ROD_DIA)**2) / (4 * Fex)
        Rod_Shear_FOS = (Rod_Shear * np.pi * Rod_Piston_Engage_Dia * Rod_Piston_Engage_Length) / (2 * F)
        Rod_Buckling_FOS = (np.pi**3 * E * ROD_DIA**4) / (64 * F * STROKE**2)

        # Rod Eye
        Eye_Tensile = 3300
        Eye_Shear = 3780
        Eye_Thickness_FOS = Eye_Tensile / (F / (2 * Rod_Eye_Thickness * Pin_width_RE))
        Least_Eye_FOS = Eye_Tensile / (F / (2 * Rod_Eye_Least_Thickness * Pin_width_RE))
        Eye_Shear_FOS = Eye_Shear / (Fex / (2 * Pin_width_RE * sqrt(((Pin_dia_RE/2) + Rod_Eye_Thickness)**2 - (Pin_dia_RE/2)**2)))

        # Piston
        Piston_Tensile = 3600
        Piston_Shear = 3780
        Piston_Axial_FOS = Piston_Tensile / (4 * F / (np.pi * (BORE**2 - Rod_Piston_Engage_Dia**2)))
        Piston_Shear_FOS = Piston_Shear / (F / (Piston_Length * np.pi * Rod_Piston_Engage_Dia))

        # Tube
        Tube_Tensile = 5000
        Tube_Axial_FOS = Tube_Tensile / (4 * F / (np.pi * ((BORE + 2*Thickness)**2 - BORE**2)))
        Tube_Circum_FOS = Tube_Tensile / (Working_Pressure * (BORE + 2*Thickness) / (2*Thickness))

        # CEC
        CEC_Tensile = 3300
        CEC_Shear = 3600
        CEC_Tensile_FOS = CEC_Tensile / (Fex / (2 * CEC_Thickness * Pin_width_CEC))
        CEC_Shear_FOS = CEC_Shear / (Fex / (2 * Pin_width_CEC * sqrt(((Pin_dia_CEC/2) + CEC_Thickness)**2 - (Pin_dia_CEC/2)**2)))

        # HEC
        HEC_Tensile = 3600
        HEC_Shear = 3780
        HEC_Tensile_FOS = (HEC_Tensile * np.pi * ((BORE)**2 - (HEC_Inner_dia)**2)) / (4 * F)
        HEC_Shear_FOS = (HEC_Shear * np.pi * Engage_Length * BORE) / F

        # Weld
        Tube_CECWeld_FOS = (CEC_Tube_weld_Strength * 100 * np.pi * (((BORE + 2 * Thickness)**2 - (BORE + 1)**2))) / (4 * F)

        # Port
        Port_Circum_HEC = Tube_Tensile / (Working_Pressure * (BORE + 2*Thickness) / (2*Thickness)) * (1 + Port_hole_dia_RHS / (2*Thickness))
        Port_Circum_CEC = Tube_Tensile / (Working_Pressure * (BORE + 2*Thickness) / (2*Thickness)) * (1 + port_hole_dia_near_CEC / (2*Thickness))

        # # FOS Results
        # st.text(f"Rod FOS - Axial: {Rod_Axial_FOS:.2f}, Shear: {Rod_Shear_FOS:.2f}, Buckling: {Rod_Buckling_FOS:.2f}")
        # st.text(f"Rod Eye FOS - Thickness: {Eye_Thickness_FOS:.2f}, Least: {Least_Eye_FOS:.2f}, Shear: {Eye_Shear_FOS:.2f}")
        # st.text(f"Piston FOS - Axial: {Piston_Axial_FOS:.2f}, Shear: {Piston_Shear_FOS:.2f}")
        # st.text(f"Tube FOS - Axial: {Tube_Axial_FOS:.2f}, Circumferential: {Tube_Circum_FOS:.2f}")
        # st.text(f"CEC FOS - Tensile: {CEC_Tensile_FOS:.2f}, Shear: {CEC_Shear_FOS:.2f}")
        # st.text(f"HEC FOS - Tensile: {HEC_Tensile_FOS:.2f}, Shear: {HEC_Shear_FOS:.2f}")
        # st.text(f"Weld FOS: {Tube_CECWeld_FOS:.2f}")
        # st.text(f"Port FOS - HEC: {Port_Circum_HEC:.2f}, CEC: {Port_Circum_CEC:.2f}")
        with st.container():       
            # st.markdown("### Factor of Safety (FOS) Summary")
            # Toggle this variable to control interactivity
            is_editable = True  # Set to False if you want to disable input fields
            
            # ROD FOS
            st.subheader("Rod FOS")
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Rod_Axial_FOS:", value=f"{Rod_Axial_FOS:.1f}", disabled=not is_editable)
                st.text_input("Rod_Shear_FOS:", value=f"{Rod_Shear_FOS:.1f}", disabled=not is_editable)
                st.text_input("Rod_Buckling_FOS:", value=f"{Rod_Buckling_FOS:.1f}", disabled=not is_editable)
            with col2:
                st.text_input("Desired_Rod_Tensile_FOS:", value="4.00", disabled=not is_editable)
                st.text_input("Desired_Rod_Shear_FOS:", value="7.00", disabled=not is_editable)
                st.text_input("Desired_Rod_Buckling_FOS:", value="4.00", disabled=not is_editable)
            
            # ROD EYE FOS
            st.subheader("Rod Eye FOS")
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Eye_Thickness_FOS:", value=f"{Eye_Thickness_FOS:.2f}", disabled=not is_editable)
                st.text_input("Leasteye_Thickness_FOS:", value=f"{Least_Eye_FOS:.2f}", disabled=not is_editable)
                st.text_input("Eye_Shear_FOS:", value=f"{Eye_Shear_FOS:.2f}", disabled=not is_editable)
            with col2:
                st.text_input("Desired_Eye_FOS:", value="4.00", disabled=not is_editable)
                st.text_input("Desired_Least_Eye_FOS:", value="4.00", disabled=not is_editable)
                st.text_input("Desired_Eye_Shear_FOS:", value="7.00", disabled=not is_editable)
    
            
            # PISTON FOS
            st.subheader("Piston FOS")
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Piston_Axial_FOS:", value=f"{Piston_Axial_FOS:.2f}", disabled=not is_editable)
                st.text_input("Piston_Shear_FOS:", value=f"{Piston_Shear_FOS:.2f}", disabled=not is_editable)
            with col2:
                st.text_input("Desired_Piston_Tensile_FOS:", value="4.00", disabled=not is_editable)
                st.text_input("Piston_Piston_Shear_FOS:", value="7.00", disabled=not is_editable)
            
            # TUBE FOS
            st.subheader("Tube FOS")
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Tube_Axial_FOS:", value=f"{Tube_Axial_FOS:.2f}", disabled=not is_editable)
                st.text_input("Tube_Circum_FOS:", value=f"{Tube_Circum_FOS:.2f}", disabled=not is_editable)
            with col2:
                st.text_input("Desired_Axial_FOS:", value="4.00", disabled=not is_editable)
                st.text_input("Desired_Circum_FOS:", value="2.00", disabled=not is_editable)
            
            # CEC FOS
            st.subheader("CEC FOS")
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("CEC_Tensile_FOS:", value=f"{CEC_Tensile_FOS:.2f}", disabled=not is_editable)
                st.text_input("CEC_Shear_FOS:", value=f"{CEC_Shear_FOS:.2f}", disabled=not is_editable)
    
            with col2:
                st.text_input("Desired_CEC_Tensile_FOS:", value="4.00", disabled=not is_editable)
                st.text_input("Desired_CEC_Shear_FOS:", value="7.00", disabled=not is_editable)
            
            # HEC FOS
            st.subheader("HEC FOS")
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("HEC_Tensile_FOS:", value=f"{HEC_Tensile_FOS:.2f}", disabled=not is_editable)
                st.text_input("HEC_Shear_FOS:", value=f"{HEC_Shear_FOS:.2f}", disabled=not is_editable)
            with col2:
                st.text_input("Desired_HEC_Tensile_FOS:", value="4.00", disabled=not is_editable)
                st.text_input("Desired_HEC_Shear_FOS:", value="7.00", disabled=not is_editable)
            
            # TUBE-CEC WELD FOS
            st.subheader("Tube-CEC Weld FOS")
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Tube_CECWeld_FOS:", value=f"{Tube_CECWeld_FOS:.2f}", disabled=not is_editable)
            with col2:
                st.text_input("Desired_Tube_CEC_Weld_FOS:", value="7.00", disabled=not is_editable)
            
            # PORT FOS
            st.subheader("Port FOS")
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("HECsideHole_Circum_FOS:", value=f"{Port_Circum_HEC:.2f}", disabled=not is_editable)
                st.text_input("CECsideHole_Circum_FOS:", value=f"{Port_Circum_CEC:.2f}", disabled=not is_editable)
            with col2:
                st.text_input("Desired_HECsideHole_FOS:", value="2.00", disabled=not is_editable)
                st.text_input("Desired_CECsideHole_FOS:", value="2.00", disabled=not is_editable)

    
