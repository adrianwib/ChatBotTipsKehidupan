
import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

intents = [
    {
        "tag": "greeting",
        "patterns": ["Halo", "Hai", "Halo apa kabar", "Apa kabar", "Apa yang terbaru"],
        "responses": ["Halo", "Hai", "Hai juga", "Saya baik, terima kasih", "Tidak ada yang istimewa"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Sampai jumpa", "Selamat tinggal", "Terima kasih dan selamat tinggal"],
        "responses": ["Sampai jumpa! Jangan ragu untuk kembali jika Anda membutuhkan bantuan lain waktu.", "Selamat tinggal! Semoga harimu menyenangkan dan semoga bisa membantu Anda lain waktu."]
    },
    {
        "tag": "thanks",
        "patterns": ["Terima kasih", "Makasih", "Terima kasih banyak", "Saya menghargai ini"],
        "responses": ["Sama-sama", "Tidak masalah", "Senang bisa membantu"]
    },
    {
        "tag": "about",
        "patterns": ["Ceritakan tentang dirimu", "Apa yang bisa kamu lakukan", "Deskripsi tentang dirimu"],
        "responses": ["Saya adalah chatbot yang siap membantu Anda. Apakah ada yang bisa saya bantu?", "Saya adalah chatbot yang diciptakan untuk memberikan informasi dan bantuan. Bagaimana saya bisa membantu Anda hari ini?"]
    },
    {
        "tag": "help",
        "patterns": ["Bantuan", "Saya butuh bantuan", "Dapatkah kamu membantu saya", "Apa yang sebaiknya saya lakukan"],
        "responses": ["Tentu, ada yang ingin Anda tanyakan?", "Saya siap membantu. Apa masalahnya?", "Bagaimana saya bisa membantu Anda?"]
    },
    {
        "tag": "tips_belajar",
        "patterns": ["Bagaimana cara mahir dalam suatu mata pelajaran", "Tips untuk menguasai suatu pelajaran", "Teknik belajar agar lebih memahami"],
        "responses": ["Untuk mahir dalam suatu mata pelajaran, cobalah membuat jadwal belajar dan patuhi itu...", "Latihan pemecahan masalah secara rutin dan cari bantuan jika diperlukan..."]
    },
    {
        "tag": "weight_loss_tips",
        "patterns": ["Bagaimana cara menurunkan berat badan", "Tips for effective weight loss", "Healthy ways to reduce body weight"],
        "responses": ["Menurunkan berat badan melibatkan kombinasi pola makan seimbang dan olahraga teratur...", "Gabungkan latihan aerobik dan latihan kekuatan dalam rutinitas Anda..."]
    },
    {
        "tag": "life_tips",
        "patterns": ["Tips untuk kehidupan yang lebih baik", "Cara meningkatkan kualitas hidup", "Saran untuk menjalani kehidupan yang memuaskan"],
        "responses": ["Fokus pada kesehatan fisik dan mental Anda...", "Pelajari mengelola keuangan dengan menyimpan, mengatur anggaran, dan berinvestasi dengan bijak..."]
    },
    {
        "tag": "tips_kesehatan_mental",
        "patterns": ["Bagaimana menjaga kesehatan mental", "Tips untuk kesehatan mental yang baik", "Cara merawat kesehatan jiwa"],
        "responses": ["Jaga keseimbangan antara pekerjaan dan istirahat...", "Latih meditasi dan relaksasi untuk mengurangi stres..."]
    },
    {
        "tag": "tips_komunikasi",
        "patterns": ["Bagaimana cara berkomunikasi efektif", "Tips untuk berkomunikasi dengan baik", "Cara berbicara dengan efisien"],
        "responses": ["Dengarkan dengan cermat sebelum merespons...", "Gunakan bahasa tubuh yang terbuka dan ramah..."]
    },
    {
        "tag": "tips_pengembangan_diri",
        "patterns": ["Bagaimana cara mengembangkan diri", "Tips untuk pertumbuhan pribadi", "Cara menjadi versi terbaik dari diri sendiri"],
        "responses": ["Tetaplah belajar dengan membaca buku dan mengikuti kursus...", "Kelola waktu dengan bijak dan tetap fokus pada tujuan Anda..."]
    },
    {
        "tag": "tips_organisasi",
        "patterns": ["Bagaimana cara mengatur kehidupan yang teratur", "Tips untuk menjaga kehidupan teratur", "Cara mengelola waktu dengan baik"],
        "responses": ["Gunakan kalender atau aplikasi untuk merencanakan jadwal Anda...", "Prioritaskan tugas berdasarkan urgensi dan pentingnya..."]
    },
    {
        "tag": "tips_pertemanan",
        "patterns": ["Bagaimana cara membangun persahabatan yang baik", "Tips untuk hubungan pertemanan yang kuat", "Cara menjaga pertemanan"],
        "responses": ["Jadilah pendengar yang baik saat teman berbicara...", "Bantu teman saat mereka membutuhkan bantuan..."]
    },
    {
        "tag": "introduction",
        "patterns": ["Siapa kamu", "Bolehkah kamu memperkenalkan diri", "Siapa namamu"],
        "responses": ["Halo! Saya adalah chatbot yang siap membantu Anda. Apakah ada yang bisa saya bantu?", "Saya adalah chatbot yang diciptakan untuk memberikan informasi dan bantuan. Bagaimana saya bisa membantu Anda hari ini?"]
    },
    {
        "tag": "about_chatbot",
        "patterns": ["Ceritakan tentang dirimu", "Apa yang bisa kamu lakukan", "Deskripsi tentang dirimu"],
        "responses": ["Saya adalah chatbot cerdas yang dirancang untuk memberikan informasi dan saran...", "Saya di sini untuk membantu Anda dengan apa pun yang Anda butuhkan!"]
    }
]

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()