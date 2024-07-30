import streamlit as st
from transformers import AutoTokenizer, BartForConditionalGeneration, MarianMTModel, MarianTokenizer


def load_summarization_model():
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
    return model, tokenizer

def load_translation_models():
    models = {}
    tokenizers = {}
    language_pairs = {
        'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
        'fr-en': 'Helsinki-NLP/opus-mt-fr-en',
        'en-de': 'Helsinki-NLP/opus-mt-en-de',
        'de-en': 'Helsinki-NLP/opus-mt-de-en',
        'en-hi': 'Helsinki-NLP/opus-mt-en-hi',
        'hi-en': 'Helsinki-NLP/opus-mt-hi-en'
    }
    for pair, model_name in language_pairs.items():
        models[pair] = MarianMTModel.from_pretrained(model_name)
        tokenizers[pair] = MarianTokenizer.from_pretrained(model_name)
    return models, tokenizers

summarization_model, summarization_tokenizer = load_summarization_model()
translation_models, translation_tokenizers = load_translation_models()

def summarize_text(article):
    inputs = summarization_tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs, max_length=256, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def translate_text(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text
    language_pair = f'{source_lang}-{target_lang}'
    model = translation_models[language_pair]
    tokenizer = translation_tokenizers[language_pair]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def main():
    st.markdown(
        '''
        <style>
        h1 {
            color: #592f2f;
            font-size: 3em;
            font-weight: bold;
        }
        h2 {
            color: #f28e2c;
            font-size: 2em;
        }
        h3 {
            color: #e15759;
            font-size: 1.5em;
        }
        .stButton>button {
            background-color: #17499c;
            color: white;
            font-size: 1em;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #0a2b61;
            color: white;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

    st.title("Multilingual Text :blue[Summarizer] and :blue[Translator] :sunglasses:")

    task = st.selectbox('Choose a task', ('Summarize', 'Translate'))

    if 'summary' not in st.session_state:
        st.session_state.summary = None

    if task == 'Summarize':
        st.header("Summarize Text")
        article = st.text_area("Enter the text here:")
        if st.button("Generate Summary"):
            st.session_state.summary = summarize_text(article)
            st.subheader("Summary:")
            st.write(st.session_state.summary)

        if st.session_state.summary:
            st.header("Translate Summary")
            source_lang = st.selectbox('Source Language', ('en', 'fr', 'de', 'hi'), key='source_lang_summary')
            target_lang = st.selectbox('Translate Summary into', ('en', 'fr', 'de', 'hi'), key='target_lang_summary')
            if st.button("Translate Summary"):
                translated_summary = translate_text(st.session_state.summary, source_lang, target_lang)
                st.subheader("Translated Summary:")
                st.write(translated_summary)
        else:
            st.warning("Please generate a summary first.")

    elif task == 'Translate':
        st.header("Translate Text")
        article = st.text_area("Enter the text here:")
        source_lang = st.selectbox('Source Language', ('en', 'fr', 'de', 'hi'), key='source_lang_article')
        target_lang = st.selectbox('Translate into', ('en', 'fr', 'de', 'hi'), key='target_lang_article')
        if st.button("Translate Article"):
            translated_article = translate_text(article, source_lang, target_lang)
            st.subheader("Translated Article:")
            st.write(translated_article)

if __name__ == "__main__":
    main()
