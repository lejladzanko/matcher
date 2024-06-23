import streamlit as st
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmBlockThreshold, HarmCategory
import time

PROJECT_ID = 'qwiklabs-asl-01-8d80f58bec85'
LOCATION = 'us-east4'
vertexai.init(project=PROJECT_ID, location=LOCATION)

@st.cache_resource
def load_models():
    """
    Load the generative models for text and multimodal generation.

    Returns:
        Tuple: A tuple containing the text model and multimodal model.
    """
    text_model_pro = GenerativeModel("gemini-1.0-pro")
    multimodal_model_pro = GenerativeModel("gemini-1.0-pro-vision")
    return text_model_pro, multimodal_model_pro

def get_gemini_pro_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    for _ in range(3):  
        try:
            responses = model.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=stream,
            )

            final_response = []
            for response in responses:
                try:
                    final_response.append(response.text)
                except IndexError:
                    final_response.append("")
                    continue
            return " ".join(final_response)
        except Exception as e:
            st.warning(f"Attempt failed: {e}. Retrying...")
            time.sleep(2) 
    st.error("Service is currently unavailable. Please try again later.")
    return ""


st.markdown("<h1 style='text-align: center; color: #FF6347;'>Matcher 🔍</h1>", unsafe_allow_html=True)
text_model_pro, multimodal_model_pro = load_models()

st.write("**Matcher uses AI to search your favorite books, movies, and series**")
st.subheader("Search")

user_mood = st.selectbox(
    "How are you feeling today?",
    ["😊 Happy", "😢 Sad", "😃 Excited", "😌 Relaxed", "😐 Bored", "😟 Anxious"],
    key="user_mood"
)

tab1, tab2 = st.columns(2)

with tab1:
    st.header("Movies & Series")
    search_type = st.radio(
        "Select media type:",
        ["Movies", "Series"],
        key="search_type",
        help="Choose whether you want to search for movies or series."
    )

    search_language = st.multiselect(
        "Preferred language(s):",
        ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Other"],
        key="search_language",
        default=["English"],
        help="Select the languages you prefer for the media."
    )

    story_premise = st.multiselect(
        "Select story premises:",
        ["❤️ Love", "🏞️ Adventure", "🔍 Mystery", "😱 Horror", "😂 Comedy", "🚀 Sci-Fi", "🧙‍♂️ Fantasy", "🔪 Thriller", "🎭 Drama", "🔥 Action"],
        key="story_premise",
        default=["❤️ Love", "🏞️ Adventure"],
        help="Choose the themes you are interested in."
    )

    generate_t2t = st.button("Find", key="generate_t2t")
    if generate_t2t:
        prompt = f"""Find a {search_type.lower()} based on the following premise:\n
            - User mood: {user_mood}\n
            - Preferred language(s): {', '.join(search_language)}\n
            - Story premises: {', '.join(story_premise)}\n
            Please include the title and a brief description for each result."""
        
        with st.spinner(f"Finding {search_type.lower()}..."):
            response = get_gemini_pro_text_response(
                text_model_pro,
                prompt,
                generation_config={"temperature": 0.8, "max_output_tokens": 2048},
            )
            if response:
                st.write("### Your movie/series:")
                results = response.split("\n")
                for result in results:
                    if result:
                        st.write(result)

with tab2:
    st.header("Custom Search")
    st.write("Use the space below to describe what you are looking for:")
    custom_prompt = st.text_area("Describe your custom search", height=150, key="custom_prompt")
    generate_custom = st.button("Generate Custom Search", key="generate_custom")

    if generate_custom and custom_prompt.strip():
        with st.spinner("Generating results..."):
            response_custom = get_gemini_pro_text_response(
                text_model_pro,
                custom_prompt,
                generation_config={"temperature": 0.8, "max_output_tokens": 2048},
            )
            if response_custom:
                st.write("### Your custom search results:")
                results_custom = response_custom.split("\n")
                for result in results_custom:
                    if result:
                        st.write(result)
