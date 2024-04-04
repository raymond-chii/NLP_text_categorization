# Natural Language Processing (NLP) Projects

Welcome to my repository for ECE 467: Natural Language Processing. Here, you'll find my projects exploring various NLP techniques and algorithms.

## Project: Text Categorization using Naïve Bayes

In this project, I've implemented a text categorization system using the Naïve Bayes algorithm. The system processes articles, tokenizes the text, removes stopwords, and applies stemming before calculating the likelihood and prior probabilities. These probabilities are then used to classify the text into predefined categories.

### Features:
- **Tokenization:** Splits text into individual words, removing punctuation.
- **Stopword Removal:** Eliminates common words that do not contribute to the meaning of the text.
- **Stemming:** Reduces words to their root form.
- **Smoothing Techniques:** Implements Laplacian and Jelinek-Mercer smoothing to handle unseen words.

### Performance:
The system was tested on three different corpora, showing varying levels of accuracy. Laplacian smoothing with a constant alpha of 0.058 was chosen based on its overall performance.

### Usage:
To use this text categorization system, follow these steps:
1. Place `CHI_naive_bayes.py` in the `/TC_provided` directory of your project.
2. Run the script and follow the prompts to input the names of the training and test files.
3. The program will output a file with the predicted labels, which can be compared to the true labels for accuracy assessment.

For more detailed instructions and to view the performance results, please refer to the project documentation.
