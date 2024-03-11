import os
import string
import numpy as np
import nltk

from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('stopwords')

def read_file(base_dir, file_paths):
    files_content = {}
    for file_path in file_paths:
        full_path = os.path.join(base_dir, file_path)
        with open(full_path, 'r') as file:
            filename = os.path.basename(file_path)
            files_content[filename] = file.read().lower()
    return files_content


def read_test_list(test_list_file):
    test_files = []
    with open(test_list_file, 'r') as file:
        for line in file:
            test_files.append(line.strip())
    return test_files


def read_labels(label_file):
    labels = {}
    file_paths = []
    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            file_path = parts[0]
            filename = os.path.basename(file_path)
            labels[filename] = parts[1]
            file_paths.append(file_path)
    return labels, file_paths


def compute_vocabulary(tokenized_files):
    vocab = set()
    for tokens in tokenized_files.values():
        vocab.update(tokens)
    return vocab


def tokenize_file(files_content):
    tokens = {}
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    for filename, content in files_content.items():
        clean_content = ''.join([char for char in content if char not in string.punctuation])
        words = nltk.word_tokenize(clean_content)
        filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
        tokens[filename] = np.array(filtered_words)
    return tokens

def calculate_prior(labels):
    label_counts = Counter(labels.values())
    total_labels = len(labels)
    return {label: count / total_labels for label, count in label_counts.items()}

def calculate_likelihood(folder, vocab, labels, alpha=0.054):
    word_cond_prob = {}
    for label in set(labels.values()):
        word_cond_prob[label] = {}
        total_words = sum(folder[label].values()) + alpha * len(vocab)
        for word in vocab:
            word_cond_prob[label][word] = (folder[label].get(word, 0) + alpha) / total_words
    return word_cond_prob

def classify_article(priors, word_counts_prob, vocab, tokens):
    sum_probabilities = {}
    for label in priors.keys():
        sum_prob = np.log(priors[label])
        for word in tokens:
            if word in vocab:
                sum_prob += np.log(word_counts_prob[label].get(word, 1 / (len(vocab) + 1)))
        sum_probabilities[label] = sum_prob
    return max(sum_probabilities, key=sum_probabilities.get)

def make_predictions(priors, likelihoods, vocabulary, test_tokens_content, test_file_paths):
    predictions_list = []
    for i, (filename, tokens) in enumerate(test_tokens_content.items()):
        predicted_label = classify_article(priors, likelihoods, vocabulary, tokens)
        original_file_path = test_file_paths[i]
        predictions_list.append((original_file_path, predicted_label))
    return predictions_list


def main():
    training_labels_file = input(
        "Enter the name of the file containing the list of labeled training documents: ")
    test_list_file = input("Enter the name of the file containing the list of unlabeled test documents: ")

    base_dir = os.path.dirname(training_labels_file)
    corpus_name = os.path.basename(training_labels_file).split('_')[0]

    labels, train_file_paths = read_labels(training_labels_file)
    train_files_content = read_file(base_dir, train_file_paths)
    train_tokens_content = tokenize_file(train_files_content)
    vocabulary = compute_vocabulary(train_tokens_content)

    folder = {}
    for label in set(labels.values()):
        folder[label] = Counter()
        for filename, tokens in train_tokens_content.items():
            if labels[filename] == label:
                folder[label].update(tokens)

    priors = calculate_prior(labels)
    likelihoods = calculate_likelihood(folder, vocabulary, labels)

    test_file_paths = read_test_list(test_list_file)
    test_files_content = read_file(base_dir, test_file_paths)
    test_tokens_content = tokenize_file(test_files_content)

    predictions_list = make_predictions(priors, likelihoods, vocabulary, test_tokens_content, test_file_paths)
    output_file = os.path.join(base_dir, f"predicted_{corpus_name}_test.labels")
    with open(output_file, 'w') as file:
        for item in predictions_list:
            file_path, predicted_label = item
            file.write(f"{file_path} {predicted_label}\n")


if __name__ == '__main__':
    main()
