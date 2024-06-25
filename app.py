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
# Initialize Vertex AI
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

def get_gemini_pro_text_response(model: GenerativeModel, contents: str, generation_config: GenerationConfig, stream: bool = True):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    for attempt in range(3):  # Retry up to 3 times
        try:
            responses = model.generate_content(contents, generation_config=generation_config, safety_settings=safety_settings, stream=stream)
            final_response = []
            for response in responses:
                try:
                    final_response.append(response.text)
                except IndexError:
                    final_response.append("")
                    continue
            return " ".join(final_response)
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(2)  # Wait before retrying
    st.error("Service is currently unavailable. Please try again later.")
    return ""

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Matcher 🔍</h1>", unsafe_allow_html=True)
text_model_pro, multimodal_model_pro = load_models()

st.write("**Matcher uses AI to search your favorite books, movies, and series**")
st.subheader("Search")

# Mood check
user_mood = st.selectbox(
    "How are you feeling today?",
    ["😊 Happy", "😢 Sad", "😃 Excited", "😌 Relaxed", "😐 Bored", "😟 Anxious"],
    key="user_mood"
)

# Tabs for different media types
tab1, tab2, tab3 = st.tabs(["🎬 Movies & Series", "📚 Books", "💡 Custom Search"])

with tab1:
    st.header("Movies & Series")
    search_type = st.radio(
        "Select media type:",
        ["Movies", "Series"],
        key="search_type",
        help="Choose between searching for movies or series."
    )

    age_group = st.selectbox(
        "Select age group:",
        ["Children", "Teens", "Adults"],
        key="age_group"
    )

    search_language = st.multiselect(
        "Preferred language(s):",
        ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Other"],
        key="search_language",
        default=["English"]
    )

    search_location = st.text_input(
        "Country of origin:",
        key="search_location",
        value="USA"
    )

    story_premise = st.multiselect(
        "Select story premises (multiple selections allowed):",
        [
            "❤️ Love", "🏞️ Adventure", "🔍 Mystery", "😱 Horror", "😂 Comedy",
            "🚀 Sci-Fi", "🧙‍♂️ Fantasy", "🔪 Thriller", "🎭 Drama", "🔥 Action"
        ],
        key="story_premise",
        default=["❤️ Love", "🏞️ Adventure"]
    )

    director_author = st.text_input(
        "Favorite director (optional):",
        key="director_author",
        value=""
    )

    length_of_story = st.radio(
        "Preferred length:",
        ["Short (< 90 minutes)", "Long (> 90 minutes)"],
        key="length_of_story",
        horizontal=True,
    )

    release_year = st.slider(
        "Release year range:",
        min_value=1900, max_value=2024, value=(2000, 2024),
        key="release_year"
    )

    actors_characters = st.text_input(
        "Favorite actor or actress (optional):",
        key="actors_characters",
        value=""
    )

    time_period = st.radio(
        "Preferred time period:",
        ["Past", "Present", "Future"],
        key="time_period",
        horizontal=True,
        index=1
    )

    prompt = f"""Find a {search_type} based on the following premise:
    user_mood: {user_mood}
    age_group: {age_group}
    search_type: {search_type}
    search_language: {", ".join(search_language)}
    search_location: {search_location}
    story_premise: {", ".join(story_premise)}
    director_author: {director_author}
    length_of_story: {length_of_story}
    release_year: {release_year[0]}-{release_year[1]}
    actors_characters: {actors_characters}
    time_period: {time_period}
    """

    config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
    }

    generate_t2t = st.button("Find my favorite movie/series", key="generate_t2t")
    if generate_t2t and prompt:
        with st.spinner("Finding your favorite movie/series using Gemini 1.0 Pro ..."):
            response = get_gemini_pro_text_response(
                text_model_pro,
                prompt,
                generation_config=config,
            )
            if response:
                st.subheader("Results:")
                results = response.split("\n")
                match_found = False
                for result in results:
                    if result:
                        if not match_found:
                            st.markdown("<h2 style='text-align: center; color: #FF6347;'>You have a match!</h2>", unsafe_allow_html=True)
                            match_found = True
                        st.write(result)

with tab2:
    st.header("Books")

    # Age group selection
    age_group_books = st.selectbox(
        "Select age group:",
        ["Children", "Teens", "Adults"],
        key="age_group_books"
    )

    search_language_books = st.multiselect(
        "Preferred language(s):",
        ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Other"],
        key="search_language_books",
        default=["English"]
    )

    search_location_books = st.text_input(
        "Country of origin:",
        key="search_location_books",
        value="USA"
    )

    story_premise_books = st.multiselect(
        "Select story premises (multiple selections allowed):",
        [
            "❤️ Love", "🏞️ Adventure", "🔍 Mystery", "😱 Horror", "😂 Comedy",
            "🚀 Sci-Fi", "🧙‍♂️ Fantasy", "🔪 Thriller", "🎭 Drama", "🔥 Action"
        ],
        key="story_premise_books",
        default=["❤️ Love", "🏞️ Adventure"]
    )

    author = st.text_input(
        "Favorite author (optional):",
        key="author",
        value=""
    )

    length_of_story_books = st.radio(
        "Preferred length:",
        ["Short (< 300 pages)", "Long (> 300 pages)"],
        key="length_of_story_books",
        horizontal=True,
    )

    release_year_books = st.slider(
        "Release year range:",
        min_value=1900, max_value=2024, value=(2000, 2024),
        key="release_year_books"
    )

    characters_books = st.text_input(
        "Favorite book character (optional):",
        key="characters_books",
        value=""
    )

    time_period_books = st.radio(
        "Preferred time period:",
        ["Past", "Present", "Future"],
        key="time_period_books",
        horizontal=True,
        index=1
    )

    # Prompt for book recommendations
    prompt_books = f"""Find a book based on the following premise:
    user_mood: {user_mood}
    age_group_books: {age_group_books}
    search_language_books: {", ".join(search_language_books)}
    search_location_books: {search_location_books}
    story_premise_books: {", ".join(story_premise_books)}
    author: {author}
    length_of_story_books: {length_of_story_books}
    release_year_books: {release_year_books[0]}-{release_year_books[1]}
    characters_books: {characters_books}
    time_period_books: {time_period_books}
    """

    config_books = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
    }

    generate_books = st.button("Find my favorite book", key="generate_books")
    if generate_books and prompt_books:
        with st.spinner("Finding your favorite book using Gemini 1.0 Pro ..."):
            response_books = get_gemini_pro_text_response(
                text_model_pro,
                prompt_books,
                generation_config=config_books,
            )
            if response_books:
                st.subheader("Results:")
                results_books = response_books.split("\n")
                match_found_books = False
                for result_books in results_books:
                    if result_books:
                        if not match_found_books:
                            st.markdown("<h2 style='text-align: center; color: #FF6347;'>You have a match!</h2>", unsafe_allow_html=True)
                            match_found_books = True
                        st.write(result_books)

with tab3:
    st.header("Custom Search")
    st.write("Use the space below to describe what you are looking for:")
    custom_prompt = st.text_area("Describe your custom search", height=150, key="custom_prompt", placeholder="Enter your custom search prompt here...")

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

