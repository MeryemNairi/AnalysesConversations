import streamlit as st
import fitz  # PyMuPDF
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict
import re

# Télécharger les ressources NLTK si elles ne sont pas déjà téléchargées
nltk.download('punkt')
nltk.download('vader_lexicon')

def extract_text_from_pdf(file_contents):
    text = ""
    try:
        doc = fitz.open(stream=file_contents, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du texte du PDF : {str(e)}")
        return None
    return text

def analyze_chat(chat_text):
    if not chat_text:
        return None

    # Initialiser les variables
    sid = SentimentIntensityAnalyzer()
    user_counts = defaultdict(int)
    user_sentiments = defaultdict(float)
    sentences = nltk.sent_tokenize(chat_text)

    # Analyser chaque phrase pour le sentiment et compter les messages par utilisateur
    for sentence in sentences:
        # Extraire l'utilisateur du message en utilisant regex
        match = re.search(r'\d{2}/\d{2}/\d{2}, \d{2}:\d{2} - (.+?):', sentence)
        if match:
            user = match.group(1)
            scores = sid.polarity_scores(sentence)
            user_counts[user] += 1
            user_sentiments[user] += scores['compound']

    # Déterminer l'utilisateur le plus actif
    most_active_user = max(user_counts, key=user_counts.get)

    # Déterminer les utilisateurs les plus positifs et négatifs
    most_positive_user = max(user_sentiments, key=user_sentiments.get)
    most_negative_user = min(user_sentiments, key=user_sentiments.get)

    return most_active_user, most_positive_user, most_negative_user

def generate_conversation_context(chat_text, most_active_user):
    context = ""

    # Analyser la conversation
    _, _, _ = analyze_chat(chat_text)

    # Générer le contexte de la conversation
    context += f"La conversation est principalement dirigée par {most_active_user}. "

    # Extraire les phrases de la conversation pour déterminer le contexte
    sentences = nltk.sent_tokenize(chat_text)
    topics = set()

    for sentence in sentences:
        if "AI" in sentence or "IA" in sentence or "IA" in sentence or "intelligence artificielle" in sentence:
            topics.add("intelligence artificielle")
        if "formation" in sentence or "cours" in sentence or "PNL" in sentence:
            topics.add("formation et développement personnel")
        if "PDF" in sentence or "document" in sentence:
            topics.add("travail sur des documents")

    if topics:
        context += "Les principaux sujets de la conversation incluent : "
        context += ", ".join(topics) + ". "
    else:
        context += "Le sujet principal de la conversation n'est pas clairement défini. "

    return context

def main():
    st.title("Analyse de conversation WhatsApp")
    st.write("Téléchargez un fichier PDF contenant une conversation WhatsApp à analyser.")

    uploaded_file = st.file_uploader("Télécharger PDF", type=["pdf"])

    if uploaded_file is not None:
        file_contents = uploaded_file.read()
        chat_text = extract_text_from_pdf(file_contents)
        if chat_text is not None:
            st.text_area("Texte de la conversation", chat_text, height=400)

            st.header("Contexte de la conversation :")
            most_active_user, _, _ = analyze_chat(chat_text)
            conversation_context = generate_conversation_context(chat_text, most_active_user)
            st.write(conversation_context)

        else:
            st.warning("Aucun message valide trouvé dans le PDF.")

if __name__ == "__main__":
    main()
