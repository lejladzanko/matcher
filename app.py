import os
import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
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

    for _ in range(3):  # Retry up to 3 times
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
            time.sleep(2)  # Wait before retrying
    st.error("Service is currently unavailable. Please try again later.")
    return ""

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #FFA500;'>StoryFinder</h1>", unsafe_allow_html=True)
text_model_pro, multimodal_model_pro = load_models()

st.write("**StoryFinder uses AI to search your favorite books, movies, and series**")
st.subheader("Search")

# Mood check
user_mood = st.selectbox(
    "How are you feeling today?",
    ["Happy", "Sad", "Excited", "Relaxed", "Bored", "Anxious"],
    key="user_mood"
)

# Search input
search_type = st.radio(
    "Select media type:",
    ["Movies", "Series", "Books"],
    key="search_type",
    horizontal=True,
)

search_language = st.multiselect(
    "What language(s)?",
    ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Other"],
    key="search_language",
)

search_location = st.text_input(
    "From which country?",
    key="search_location",
    value="Spain",
)

story_premise = st.multiselect(
    "What is the story premise? (can select multiple)",
    [
        "Love", "Adventure", "Mystery", "Horror", "Comedy",
        "Sci-Fi", "Fantasy", "Thriller", "Drama", "Action"
    ],
    key="story_premise",
    default=["Love", "Adventure"],
)

director_author = st.text_input(
    "Director/Author (optional)",
    key="director_author",
    value=""
)

length_of_story = st.radio(
    "Select the length of the movie/series:",
    ["Short", "Long"],
    key="length_of_story",
    horizontal=True,
)

# Other prompt ideas
release_year = st.slider(
    "Select the release year:",
    min_value=1900, max_value=2024, value=(2000, 2024),
    key="release_year"
)

actors_characters = st.text_input(
    "Favorite actor/actress or book character (optional):",
    key="actors_characters",
    value=""
)

max_output_tokens = 2048

prompt = f"""Find a {search_type} based on the following premise:
user_mood: {user_mood}
search_type: {search_type}
search_language: {", ".join(search_language)}
search_location: {search_location}
story_premise: {", ".join(story_premise)}
director_author: {director_author}
length_of_story: {length_of_story}
release_year: {release_year[0]}-{release_year[1]}
actors_characters: {actors_characters}

Please include the title, a brief description, and if available, the URL for the poster image for each result.
"""

config = {
    "temperature": 0.8,
    "max_output_tokens": 2048,
}

generate_t2t = st.button("Find my favorite media", key="generate_t2t")
if generate_t2t and prompt:
    with st.spinner("Finding your favorite media using Gemini 1.0 Pro ..."):
        response = get_gemini_pro_text_response(
            text_model_pro,
            prompt,
            generation_config=config,
        )
        if response:
            st.write("### Your media:")
            results = response.split("\n")
            for result in results:
                if result:
                    st.write(result)
                    if "URL for the poster image" in result:
                        poster_url = result.split("URL for the poster image: ")[-1]
                        st.image(poster_url, width=300)
