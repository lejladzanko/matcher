import streamlit as st
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmBlockThreshold, HarmCategory
import time

# Initialize Vertex AI (if applicable, replace with your specific initialization)
# PROJECT_ID = 'your-project-id'
# LOCATION = 'us-east4'
# vertexai.init(project=PROJECT_ID, location=LOCATION)

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
st.markdown("<h1 style='text-align: center; color: #FF6347;'>🔍 Matcher App</h1>", unsafe_allow_html=True)
st.write("Welcome to Matcher App, powered by AI to help you find your favorite books, movies, and series!")
st.write("Choose your mood, preferences, and let Matcher suggest personalized recommendations.")
st.write("---")

# Load models
text_model_pro, multimodal_model_pro = load_models()

# Mood check
user_mood = st.selectbox(
    "How are you feeling today?",
    ["😊 Happy", "😢 Sad", "😃 Excited", "😌 Relaxed", "😐 Bored", "😟 Anxious"],
    key="user_mood"
)

# Tabs for different media types
tab1, tab2, tab3 = st.columns(3)

with tab1:
    st.header("Movies & Series")
    search_type_movies = st.radio(
        "Select media type:",
        ["Movies", "Series"],
        key="search_type_movies",
        help="Choose whether you want to search for movies or series."
    )

    search_language_movies = st.multiselect(
        "Preferred language(s):",
        ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Other"],
        key="search_language_movies",
        default=["English"],
        help="Select the languages you prefer for the media."
    )

    story_premise_movies = st.multiselect(
        "Select story premises:",
        ["❤️ Love", "🏞️ Adventure", "🔍 Mystery", "😱 Horror", "😂 Comedy", "🚀 Sci-Fi", "🧙‍♂️ Fantasy", "🔪 Thriller", "🎭 Drama", "🔥 Action"],
        key="story_premise_movies",
        default=["❤️ Love", "🏞️ Adventure"],
        help="Choose the themes you are interested in."
    )

    generate_movies = st.button("Find", key="generate_movies")
    if generate_movies:
        prompt_movies = f"""Find a {search_type_movies.lower()} based on the following premise:\n
            - User mood: {user_mood}\n
            - Preferred language(s): {', '.join(search_language_movies)}\n
            - Story premises: {', '.join(story_premise_movies)}\n
            Please include the title and a brief description for each result."""
        
        with st.spinner(f"Finding {search_type_movies.lower()}..."):
            response_movies = get_gemini_pro_text_response(
                text_model_pro,
                prompt_movies,
                generation_config={"temperature": 0.8, "max_output_tokens": 2048},
            )
            if response_movies:
                st.write("### Your movie/series:")
                results_movies = response_movies.split("\n")
                for result in results_movies:
                    if result:
                        st.write(result)

with tab2:
    st.header("Books")
    search_language_books = st.multiselect(
        "Preferred language(s):",
        ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Other"],
        key="search_language_books",
        default=["English"],
        help="Select the languages you prefer for the books."
    )

    story_premise_books = st.multiselect(
        "Select story premises:",
        ["❤️ Love", "🏞️ Adventure", "🔍 Mystery", "😱 Horror", "😂 Comedy", "🚀 Sci-Fi", "🧙‍♂️ Fantasy", "🔪 Thriller", "🎭 Drama", "🔥 Action"],
        key="story_premise_books",
        default=["❤️ Love", "🏞️ Adventure"],
        help="Choose the themes you are interested in for books."
    )

    generate_books = st.button("Find", key="generate_books")
    if generate_books:
        prompt_books = f"""Find a book based on the following premise:\n
            - User mood: {user_mood}\n
            - Preferred language(s): {', '.join(search_language_books)}\n
            - Story premises: {', '.join(story_premise_books)}\n
            Please include the title and a brief description for each result."""
        
        with st.spinner("Finding books..."):
            response_books = get_gemini_pro_text_response(
                text_model_pro,
                prompt_books,
                generation_config={"temperature": 0.8, "max_output_tokens": 2048},
            )
            if response_books:
                st.write("### Your books:")
                results_books = response_books.split("\n")
                for result in results_books:
                    if result:
                        st.write(result)

with tab3:
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
