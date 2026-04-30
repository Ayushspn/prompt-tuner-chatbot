from dotenv import load_dotenv
import os
import streamlit as st
from openai import OpenAI
from utils import build_prompt

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI()

st.title("Prompt Tuner Chatbot 🎯")

role = st.text_input("Role (e.g., Resume Writer, Teacher)")
goal = st.text_input("Goal (e.g., Rewrite confidently, Explain simply)")
examples = st.text_area("Few-shot examples")
user_input = st.text_area("Your query")

temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7)

if st.button("Generate Response"):
    prompt = build_prompt(role, goal, examples, user_input)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )
    st.write("### Model Response:")
    st.write(response.choices[0].message.content)