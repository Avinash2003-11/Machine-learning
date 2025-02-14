import streamlit as st
st.title('Login page')
st.write('Username')
username = st.text_input('Enter your Username', key = "user_input")
st.write('password')
password = st.text_input('Enter your Password', type = 'password', key= 'pass_input')
col1, col2 = st.columns([1,1])
col1.button('login')
col2.button("cancel")