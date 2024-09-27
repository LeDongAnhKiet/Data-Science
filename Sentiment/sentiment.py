import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import nltk

# Tải xuống các gói cần thiết cho NLTK
nltk.download('punkt')

# Tải danh sách từ dừng (stopwords)
stopwords = pd.read_csv('data/stop_words.csv', header=None)[0].tolist()

# Tải danh sách từ tích cực và tiêu cực
positive_words = set(open('data/positive-words.txt').read().splitlines())
negative_words = set(open('data/negative-words.txt').read().splitlines())

# Tải nội dung từ các tệp văn bản
with open('data/austen.txt', 'r', encoding='utf-8') as file:
    text_austen = file.readlines()

with open('data/dickens.txt', 'r', encoding='utf-8') as file:
    text_dickens = file.readlines()

# Chuẩn bị DataFrame
df = pd.DataFrame({
    'text': text_austen + text_dickens,
    'book': ['Austen'] * len(text_austen) + ['Dickens'] * len(text_dickens)
})

# Hàm xử lý văn bản: phân tách các từ và tạo các cột cho số dòng và chương
def process_text(df):
    df['line_number'] = df.groupby('book').cumcount() + 1
    df['chapter'] = df['text'].apply(lambda x: re.search(r'^chapter [\divxlc]', x, re.I) is not None)
    df['chapter'] = df['chapter'].cumsum()
    df['word'] = df['text'].apply(lambda x: word_tokenize(x.lower()))
    return df.explode('word')

# Áp dụng xử lý văn bản
tidy_data = process_text(df)

# Lọc bỏ các từ dừng
tidy_data = tidy_data[~tidy_data['word'].isin(stopwords)]

# Phân tích cảm xúc sử dụng từ điển tùy chỉnh
def sentiment_analysis(word):
    if word in positive_words:
        return 1  # Từ tích cực
    elif word in negative_words:
        return -1  # Từ tiêu cực
    else:
        return 0  # Từ trung tính

# Áp dụng phân tích cảm xúc
tidy_data['sentiment'] = tidy_data['word'].apply(sentiment_analysis)

# Nhóm theo số dòng và tính tổng cảm xúc
tidy_data['index'] = tidy_data['line_number'] // 80
sentiment_summary = tidy_data.groupby(['book', 'index']).agg({'sentiment': 'sum'}).reset_index()

# Vẽ biểu đồ cảm xúc theo thời gian
plt.figure(figsize=(12, 6))
sns.barplot(x='index', y='sentiment', hue='book', data=sentiment_summary, palette='coolwarm')
plt.title('Cảm xúc theo thời gian trong các cuốn sách')
plt.xlabel('Đoạn (Mỗi đoạn có 80 dòng)')
plt.ylabel('Điểm cảm xúc')
plt.legend(title='Cuốn sách')
plt.show()

# Tạo biểu đồ từ đám mây
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tidy_data['word']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Biểu đồ từ đám mây')
plt.show()
