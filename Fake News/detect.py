import joblib

# Tải mô hình và vectorizer đã lưu
model = joblib.load('model.pkl')
transform = joblib.load('tfidf_vectorizer.pkl')

# Dữ liệu mới để dự đoán
new_data = ["Meta’s misinformation problem has local election officials struggling to get out the truth",
            "OpenAI considering restructuring to for-profit, CTO Mira Murati and two top research execs depart", 
            "Why the NFL has TV executives freaking out over 2029", 
            "World's longest-serving death row inmate acquitted"]

# Chuyển đổi dữ liệu mới bằng cách sử dụng cùng một vectorizer
tfidf_new_data = transform.transform(new_data)

# Dự đoán
new_predictions = model.predict(tfidf_new_data)

# Xuất kết quả dự đoán
for text, prediction in zip(new_data, new_predictions):
    print(f'{text} => Predicted label: {prediction}')
