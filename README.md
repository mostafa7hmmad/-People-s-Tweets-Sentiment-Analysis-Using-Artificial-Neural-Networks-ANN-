# Twitter Sentiment Analysis using ANN Techniques 🚀  
![data](1.jpg)
## 📌 Project Overview  
This project applies **Artificial Neural Networks (ANN)** to analyze sentiments in people's tweets. It involves **data collection, preprocessing, model training, and deployment as a web app using Streamlit**. The model classifies tweets into three sentiment categories:  
🔴 **Negative** | 🟡 **Neutral** | 🟢 **Positive**  

---

## 📊 Data Processing Pipeline  
### **1️⃣ Data Collection**  
- Dataset is collected from Twitter, containing real-world tweets with labels for sentiment classification.  

### **2️⃣ Data Exploration**  
- Analyzing tweet distribution, most frequent words, and sentiment balance.  
- Visualizing the dataset with graphs and word clouds.  

### **3️⃣ Data Preprocessing**  
- **Downloading stopwords** for text cleaning.  
- **Text Cleaning**: Removing URLs, special characters, and non-alphabetic text.  
- **Normalizing Letters**: Converting all text to lowercase.  
- **Removing Stopwords**: Filtering out unimportant words.  

### **4️⃣ Feature Engineering**  
- **Tokenization**: Converting text into numerical sequences.  
- **Convert Tokenized Texts to Sequences** using Keras Tokenizer.  
- **Padding Sequences**: Ensuring uniform input length for the model.  

---

## 🧠 ANN Model Architecture  
- **Input Layer**: Text sequences converted to embeddings.  
- **Hidden Layers**: Three fully connected layers with ReLU activation.  
- **Output Layer**: Softmax activation for multi-class sentiment classification.  

### **🔧 Model Compilation & Training**  
- **Loss Function**: `sparse_categorical_crossentropy`  
- **Optimizer**: `adam`  
- **Metrics**: `accuracy`  

---

## 🎯 Model Evaluation  
- **Training Accuracy**: ✅ ~94%  
- **Test Accuracy**: ⚠️ ~63% (Could be improved with hyperparameter tuning)  
- **Confusion Matrix & Classification Report** for performance assessment.  

---

## 🌐 Deployment as a Web App  
The model is deployed using **Streamlit**, providing a user-friendly interface for sentiment analysis.  

### **🔄 Steps to Run the App**  
1. Install dependencies:  
   ```sh
   pip install -r requirements.txt
