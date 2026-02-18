 # Chinese Cyberbullying Detection Model

**A Machine Learning Portfolio Project for University Applications**

This project is an exploration into the application of Natural Language Processing (NLP) to address the pressing social issue of online harassment. It features a deep learning model, built from scratch using PyTorch, to detect and classify cyberbullying content in Chinese text.

## 1. Project Motivation

In today's digital age, social media has become an integral part of communication. However, it has also given rise to a significant increase in cyberbullying, which can have profound negative impacts on individuals' mental health. I was motivated to start this project to understand if I could use my growing skills in programming and artificial intelligence to create a tool that could automatically identify harmful language. My goal was not just to build a functional model, but also to learn deeply about the challenges of NLP and contribute a small piece to fostering a safer and more positive online environment.

## 2. Project Overview

This project implements a complete machine learning pipeline for text classification. The core of the system is a Bidirectional Long Short-Term Memory (Bi-LSTM) network, a type of recurrent neural network well-suited for understanding sequential data like text.

The system performs the following key functions:
-   **Data Ingestion & Preprocessing**: Cleans and prepares raw Chinese text for the model.
-   **Model Training**: Trains the Bi-LSTM network on a labeled dataset of text samples.
-   **Performance Evaluation**: Measures the model's accuracy and loss on a separate validation set.
-   **Real-time Prediction**: Provides an interface to classify new, unseen sentences.

## 3. Technical Implementation

### a. Data Preprocessing
The model's performance heavily relies on the quality of the input data. The preprocessing pipeline includes:
-   **Text Cleaning**: Converting text to lowercase and removing punctuation.
-   **Chinese Word Segmentation**: Using the `jieba` library to segment sentences into individual words, a crucial step for the Chinese language.
-   **Tokenization**: Employing the `BertTokenizer` from the Hugging Face library to convert words into numerical IDs based on a large, pre-existing vocabulary.

### b. Model Architecture
The neural network is defined in the `SimpleTextClassifier` class and consists of three main layers:
1.  **Embedding Layer**: This layer transforms the numerical token IDs into dense vector representations (`embedding_dim=256`). These vectors capture the semantic meaning of the words.
2.  **Bidirectional LSTM Layer**: This is the heart of the model. It processes the text sequence in both forward and backward directions, allowing it to learn the context of each word based on what comes before and after it.
3.  **Fully Connected (Linear) Layer**: This final layer takes the output from the LSTM and makes the final classification decision (Class 0: Not Cyberbullying, Class 1: Cyberbullying).

## 4. Technology Stack

-   **Language**: Python 3
-   **Core Framework**: PyTorch
-   **NLP Libraries**: Transformers (`BertTokenizer`), Jieba
-   **Data Manipulation**: NumPy, Python's `csv` module
-   **Utilities**: Scikit-learn (for data splitting), TQDM (for progress bars)

## 5. How to Run This Project

### a. Prerequisites
-   Python 3.7+
-   An environment manager like `venv` or `conda`.

### b. Setup and Installation
1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/avoma66/Chinese-Cyberbullying-Detection.git
    cd Chinese-Cyberbullying-Detection
    ```

2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\Scripts\activate    # On Windows
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### c. Running the Model
To train the model and see predictions, simply run the main script. Make sure your dataset (`Book1.csv`) is placed inside the `data/` directory.
```bash
python main.py
```

## 6. Challenges & What I Learned

This project was a significant learning experience for me.
-   **Challenges**: The most significant technical challenge I faced was a persistent `circular import` error related to the `pandas` library. After many attempts at debugging, I learned the importance of dependency management and ultimately re-engineered the data loading module to use Python's native `csv` library, which made the code more robust and lightweight. This taught me a valuable lesson in problem-solving: sometimes the most elegant solution is the simplest one.

-   **What I Learned**: Through this project, I moved beyond theoretical knowledge and gained practical, hands-on experience in building a neural network with PyTorch. I learned the complete lifecycle of a machine learning project, from data cleaning to model evaluation. Most importantly, it solidified my passion for using technology to tackle meaningful, real-world problems.

## 7. Future Directions

I believe this project has a strong foundation that can be built upon. Potential next steps include:
-   **Enhancing the Model**: I plan to integrate a pre-trained language model like BERT as the core of the classifier, which should significantly improve its accuracy and understanding of nuanced language.
-   **Expanding the Dataset**: A larger and more diverse dataset would help the model generalize better to different types of online speech.
-   **Creating a Web Demo**: I would like to deploy the model using a framework like Flask or Gradio to create a simple web interface where anyone can test the classifier.
