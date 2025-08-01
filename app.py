import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords, sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

nltk.download('stopwords')

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/sentiwordnet')
except nltk.downloader.DownloadError:
    nltk.download('sentiwordnet')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except nltk.downloader.DownloadError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')


# --- Text Cleaning Functions (from previous cells) ---
def remove_URL(text):
    if text is not None and isinstance(text, str):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', text)
    else:
        return text

def remove_html(text):
    if text is not None and isinstance(text, str):
        html = re.compile(r'<.*?>')
        return html.sub(r'', text)
    else:
        return text

def remove_symbols(text):
    if text is not None and isinstance(text, str):
        symbols = re.compile(r'[^a-zA-Z0-9\s]')
        return symbols.sub(r'', text)
    else:
        return text

def remove_numbers(text):
    if text is not None and isinstance(text, str):
        numbers = re.compile(r'\d+')
        return numbers.sub('', text)
    else:
        return text

# --- Data Preprocessing Functions (from previous cells) ---
def case_folding(text):
    if text is not None and isinstance(text, str):
        return text.lower()
    else:
        return text

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    if text is not None and isinstance(text, str):
        words = text.split()
        filtered_text = [word for word in words if word not in stop_words]
        return ' '.join(filtered_text)
    else:
        return text

def tokenization(text):
    if text is not None and isinstance(text, str):
        return word_tokenize(text)
    else:
        return text

stemmer = PorterStemmer()
def stem_text(tokens):
    if tokens is not None and isinstance(tokens, list):
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return stemmed_tokens
    else:
        return tokens

def tokens_to_sentence(tokens):
    if tokens is not None and isinstance(tokens, list):
        return ' '.join(tokens)
    else:
        return tokens

# --- Sentiment Analysis Function (using SentiWordNet) ---
def get_word_pos(word):
    tag = nltk.pos_tag([word])[0][1].upper()
    tag_dict = {'N': 'n', 'V': 'v', 'R': 'r', 'A': 'a'}
    return tag_dict.get(tag[0], 'n')

def analyze_sentiment_sentiwordnet(text, neg_threshold=0, neutral_threshold=0.02):
    words = nltk.word_tokenize(text)
    sentiment_score = 0
    num_synsets = 0

    for word in words:
        synsets = list(swn.senti_synsets(word, get_word_pos(word)))
        if synsets:
            synset = synsets[0]
            sentiment_score += synset.pos_score() - synset.neg_score()
            num_synsets += 1

    if num_synsets > 0:
        average_sentiment = sentiment_score / num_synsets
    else:
        average_sentiment = 0

    if average_sentiment >= neutral_threshold:
        return 'Positive'
    elif average_sentiment <= -neg_threshold:
        return 'Negative'
    else:
        return 'Neutral'

# --- Function to perform all preprocessing steps ---
def preprocess_text(text):
    cleaned = remove_URL(text)
    cleaned = remove_html(cleaned)
    cleaned = remove_symbols(cleaned)
    cleaned = remove_numbers(cleaned)
    cased = case_folding(cleaned)
    stopwords_removed = remove_stopwords(cased)
    tokenized = tokenization(stopwords_removed)
    stemmed = stem_text(tokenized)
    stemmed_sentence = tokens_to_sentence(stemmed)
    return stemmed_sentence

# --- Streamlit App ---
st.title("Aplikasi Analisis Sentimen dan Klasifikasi Komentar")

# --- Feature 1: Sentiment Analysis (SentiWordNet) ---
st.header("Fitur 1: Analisis Sentimen dengan SentiWordNet")
user_text = st.text_area("Masukkan teks untuk analisis sentimen:")

if st.button("Analisis Sentimen"):
    if user_text:
        preprocessed_text = preprocess_text(user_text)
        sentiment = analyze_sentiment_sentiwordnet(preprocessed_text)
        st.write(f"Teks yang diproses: {preprocessed_text}")
        st.write(f"Hasil Analisis Sentimen: **{sentiment}**")
    else:
        st.warning("Mohon masukkan teks terlebih dahulu.")

st.markdown("---") # Separator

# --- Feature 2: Data Labeling, TF-IDF, SMOTE, Splitting, SVM Classification ---
st.header("Fitur 2: Klasifikasi Sentimen dari Data Berlabel")
st.write("Unggah file CSV yang berisi data komentar dan label sentimen ('comment', 'sentiment').")

uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    try:
        df_labeled = pd.read_csv(uploaded_file)

        # Validate columns
        if 'comment' not in df_labeled.columns or 'sentiment' not in df_labeled.columns:
            st.error("File CSV harus memiliki kolom 'comment' dan 'sentiment'.")
        else:
            st.success("File berhasil diunggah dan divalidasi.")
            st.write("Preview Data:")
            st.dataframe(df_labeled.head())

            # Ensure 'comment' column is treated as string for preprocessing
            df_labeled['comment'] = df_labeled['comment'].astype(str)

            # Apply preprocessing to the comment column
            with st.spinner("Melakukan preprocessing data..."):
                df_labeled['processed_comment'] = df_labeled['comment'].apply(preprocess_text)
            st.success("Preprocessing data selesai.")
            st.write("Preview Data setelah Preprocessing:")
            st.dataframe(df_labeled[['comment', 'processed_comment', 'sentiment']].head())


            # TF-IDF
            st.subheader("TF-IDF")
            with st.spinner("Menerapkan TF-IDF..."):
                tfidf_vectorizer = TfidfVectorizer(max_features=6000) # Use the same max_features
                X = tfidf_vectorizer.fit_transform(df_labeled['processed_comment'])
                y = df_labeled['sentiment']
            st.success("TF-IDF selesai.")
            st.write(f"Bentuk matriks TF-IDF: {X.shape}")

            # SMOTE (Optional based on data imbalance)
            st.subheader("SMOTE (Resampling Data)")
            st.write(f"Distribusi kelas sebelum SMOTE: {Counter(y)}")
            # Check for imbalance
            sentiment_counts = y.value_counts()
            min_class_size = sentiment_counts.min()
            total_samples = sentiment_counts.sum()
            # A simple heuristic for checking imbalance (e.g., if smallest class is less than 10% of total)
            if min_class_size < total_samples * 0.1 and len(sentiment_counts) > 1:
                st.info("Data menunjukkan ketidakseimbangan kelas. Menerapkan SMOTE.")
                with st.spinner("Menerapkan SMOTE..."):
                    smote = SMOTE(random_state=42)
                    X_res, y_res = smote.fit_resample(X, y)
                st.success("SMOTE selesai.")
                st.write(f"Distribusi kelas setelah SMOTE: {Counter(y_res)}")
                st.write(f"Bentuk data setelah SMOTE: {X_res.shape}")

                # Visualize SMOTE result
                smote_counts = pd.DataFrame.from_dict(Counter(y_res), orient='index').reset_index()
                smote_counts.columns = ['Sentiment', 'Count']
                total_comments_smote = smote_counts['Count'].sum()
                smote_counts['Percentage'] = (smote_counts['Count'] / total_comments_smote) * 100

                fig1, ax1 = plt.subplots(figsize=(8, 6))
                sns.barplot(x='Sentiment', y='Count', hue='Sentiment', data=smote_counts, palette='viridis', legend=False, ax=ax1)
                ax1.set_xlabel("Sentimen")
                ax1.set_ylabel("Jumlah Komentar")
                ax1.set_title("Distribusi Sentimen Komentar Setelah SMOTE")
                for index, row in smote_counts.iterrows():
                    ax1.text(index, row['Count'] + 5, f"{row['Count']}\n({row['Percentage']:.1f}%)", color='black', ha="center")
                ax1.set_ylim(0, smote_counts['Count'].max() * 1.2)
                st.pyplot(fig1)

            else:
                st.info("Data relatif seimbang atau hanya memiliki satu kelas. SMOTE tidak diterapkan.")
                X_res, y_res = X, y # Use original data if SMOTE is not applied


            # Data Splitting
            st.subheader("Data Splitting (Training and Testing)")
            with st.spinner("Melakukan data splitting..."):
                X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res if len(np.unique(y_res)) > 1 else None)
            st.success("Data splitting selesai.")
            st.write(f"Bentuk X_train: {X_train.shape}")
            st.write(f"Bentuk X_test: {X_test.shape}")
            st.write(f"Bentuk y_train: {y_train.shape}")
            st.write(f"Bentuk y_test: {y_test.shape}")
            st.write(f"Distribusi kelas di data training: {Counter(y_train)}")
            st.write(f"Distribusi kelas di data testing: {Counter(y_test)}")

            # Visualize Data Splitting
            split_counts = pd.DataFrame({
                'Set Data': ['Data Latih', 'Data Uji'],
                'Jumlah Data': [X_train.shape[0], X_test.shape[0]]
            })

            fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot 1: Jumlah Data Latih dan Uji
            bars_split = sns.barplot(x='Set Data', y='Jumlah Data', hue='Set Data', data=split_counts, palette='pastel', legend=False, ax=ax2)
            ax2.set_xlabel("Set Data")
            ax2.set_ylabel("Jumlah Data")
            ax2.set_title("Distribusi Data Latih dan Data Uji")
            total_data = split_counts['Jumlah Data'].sum()
            for index, row in split_counts.iterrows():
                percentage = (row['Jumlah Data'] / total_data) * 100
                ax2.text(index, row['Jumlah Data'] + 5, f"{row['Jumlah Data']}\n({percentage:.1f}%)", color='black', ha="center")
            ax2.set_ylim(0, split_counts['Jumlah Data'].max() * 1.2)

            # Plot 2: Distribusi Kelas pada Data Latih dan Uji
            train_dist = pd.DataFrame.from_dict(Counter(y_train), orient='index', columns=['Jumlah']).reset_index()
            train_dist.columns = ['Sentimen', 'Jumlah']
            train_dist['Set Data'] = 'Data Latih'

            test_dist = pd.DataFrame.from_dict(Counter(y_test), orient='index', columns=['Jumlah']).reset_index()
            test_dist.columns = ['Sentimen', 'Jumlah']
            test_dist['Set Data'] = 'Data Uji'

            combined_dist = pd.concat([train_dist, test_dist])

            bars_class = sns.barplot(x='Sentimen', y='Jumlah', hue='Set Data', data=combined_dist, palette='viridis', ax=ax3)
            ax3.set_xlabel("Sentimen")
            ax3.set_ylabel("Jumlah Data")
            ax3.set_title("Distribusi Kelas pada Data Latih dan Data Uji")
            for container in bars_class.containers:
                 ax3.bar_label(container)

            plt.tight_layout()
            st.pyplot(fig2)


            # Support Vector Machine (SVM)
            st.subheader("Pemodelan Support Vector Machine (SVM)")
            with st.spinner("Melatih model SVM..."):
                svm_model = SVC(kernel='rbf', random_state=42)
                svm_model.fit(X_train, y_train)
            st.success("Model SVM terlatih.")

            # Evaluate SVM Model
            st.subheader("Evaluasi Model SVM")
            with st.spinner("Mengevaluasi model SVM..."):
                 y_pred_svm = svm_model.predict(X_test)

                 accuracy_svm = accuracy_score(y_test, y_pred_svm)
                 st.write(f"Akurasi model SVM: {accuracy_svm:.4f}")

                 st.write("Laporan Klasifikasi SVM:")
                 st.text(classification_report(y_test, y_pred_svm, zero_division=0))

                 conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
                 st.write("Confusion Matrix SVM:")
                 st.text(conf_matrix_svm)

                 # Visualisasi Confusion Matrix
                 fig3, ax4 = plt.subplots(figsize=(8, 6))
                 sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues',
                             xticklabels=svm_model.classes_, yticklabels=svm_model.classes_, ax=ax4)
                 ax4.set_xlabel('Prediksi')
                 ax4.set_ylabel('Aktual')
                 ax4.set_title('Confusion Matrix SVM')
                 st.pyplot(fig3)

                 # Confusion Matrix in Percentage
                 conf_matrix_svm_percentage = conf_matrix_svm.astype('float') / conf_matrix_svm.sum(axis=1)[:, np.newaxis]
                 st.write("Confusion Matrix SVM (dalam Persentase):")
                 st.text(conf_matrix_svm_percentage)

                 fig4, ax5 = plt.subplots(figsize=(8, 6))
                 sns.heatmap(conf_matrix_svm_percentage, annot=True, fmt='.2%', cmap='Blues',
                             xticklabels=svm_model.classes_, yticklabels=svm_model.classes_, ax=ax5)
                 ax5.set_xlabel('Prediksi')
                 ax5.set_ylabel('Aktual')
                 ax5.set_title('Confusion Matrix SVM (dalam Persentase)')
                 st.pyplot(fig4)

            st.success("Evaluasi model SVM selesai.")


    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
