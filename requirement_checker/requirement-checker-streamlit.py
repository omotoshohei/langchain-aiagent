import streamlit as st
import openai

# Constants
AI_MODEL = "gpt-4o"
TOKEN_COUNT = 4096
MAX_USES = 3

# Set page configuration
st.set_page_config(page_title="Requirements Checker", page_icon=":bar_chart:")

# Load API key and prompt from secrets
openai.api_key = st.secrets['OPENAI_API_KEY']
prompt_template = st.secrets['PROMPT_REQUIREMENT']

# Page title
st.title('Requirements Checker')

# Styling (optional)
st.markdown(
"""
<style>
/* Custom style adjustments */
.st-emotion-cache-iiif1v { display: none !important; }
.st-emotion-cache-13ln4jf {padding: 6rem 1rem 0rem;}
@media (max-width: 50.5rem) {
.st-emotion-cache-13ln4jf {
max-width: calc(0rem + 100vw);
}
}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize usage counter and language in session state
if 'usage_count' not in st.session_state:
    st.session_state['usage_count'] = 0
if 'language' not in st.session_state:
    st.session_state['language'] = 'English'

def generate_requirements(language, instruction, background):
    """Generates requirements based on the given instructions and background context."""
    if st.session_state['usage_count'] < MAX_USES:
        st.session_state['usage_count'] += 1  # Increment the usage counter
        task_prompt = f"""
        - Task: {prompt_template}
        - Output language: {language}
        - Instruction from your client: {instruction}
        - Background: {background}
        """
        with st.spinner('Defining requirements...'):
            response = openai.ChatCompletion.create(
                model=AI_MODEL,
                messages=[{"role": "user", "content": task_prompt}],
                max_tokens=TOKEN_COUNT
            )
        return response["choices"][0]["message"]["content"]
    else:
        st.error("You have reached your maximum usage limit.")
        return None

# Determine button text based on current language
if st.session_state['language'] == 'English':
    switch_button_text = 'Japanese（日本語）'
else:
    switch_button_text = 'English'

# Language switcher button
if st.button(switch_button_text):
    if st.session_state['language'] == 'English':
        st.session_state['language'] = 'Japanese'
    else:
        st.session_state['language'] = 'English'
    st.experimental_rerun()

# Display form based on selected language
if st.session_state['language'] == 'English':
    st.subheader('English')
    en_input_instruction = st.text_input(
        "Enter the instruction from your boss or client (e.g., Let's actively push this new product into the market. Everyone, let's brainstorm ideas.)",
        key="en_input_instruction"
    )
    en_input_background = st.text_input(
        "Background (e.g., Since the company is struggling with sales, we want to start selling a new product.)",
        key="en_input_background"
    )
    if st.button("Define the requirements", key="en_define"):
        result = generate_requirements("English", en_input_instruction, en_input_background)
        if result:
            st.write(result)
else:
    st.subheader('日本語')
    ja_input_instruction = st.text_input(
        "クライアントからの指示を入力ください (例：「この新商品、市場にどんどん押し出していこう。みんなでアイディア出して。」)",
        key="ja_input_instruction"
    )
    ja_input_background = st.text_input(
        "背景を入力ください (例：「会社が売り上げに伸び悩んでいるから、新しい商品を売っていきたい。」)",
        key="ja_input_background"
    )
    if st.button("要件定義をする", key="ja_define"):
        result = generate_requirements("Japanese", ja_input_instruction, ja_input_background)
        if result:
            st.write(result)

