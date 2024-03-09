import os
import string
import numpy as np
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')

def read_file(base_dir, file_paths):
    files_content = {}
    for file_path in file_paths:
        # Prepend the base directory to the file path
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

def tokenize_file(files_content):
    tokens = {}
    stop_words = set(stopwords.words('english'))
    for filename, content in files_content.items():
        clean_content = ''.join([char for char in content if char not in string.punctuation])
        words = nltk.word_tokenize(clean_content)
        filtered_words = [word for word in words if word not in stop_words]
        tokens[filename] = np.array(filtered_words)
    return tokens

def build_vocabulary(tokenized_files):
    vocabulary = set()
    for tokens in tokenized_files.values():
        vocabulary.update(tokens)
    return vocabulary

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

def calculate_prior(labels):
    label_counts = {}
    priors = {}
    for label in labels.values():
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    total_labels = len(labels)
    for label, count in label_counts.items():
        priors[label] =  count / total_labels
    return priors

def calculate_likelihood(tokens, vocab, labels):
    word_counts = {}
    class_word_counts = {}
    for label in set(labels.values()):
        word_counts[label] = {}
        for word in vocab:
            word_counts[label][word] = 0

    for label in set(labels.values()):
        class_word_counts[label] = 0

    for filename, tokens in tokens.items():
        label = labels[filename]
        for token in tokens:
            word_counts[label][token] += 1
            class_word_counts[label] += 1
    vocab_size = len(vocab)
    word_counts_prob = {}
    for label in word_counts:
        word_counts_prob[label] = {}
    for label, words in word_counts.items():
        total_words = class_word_counts[label] + vocab_size
        for word, count in words.items():
            word_counts_prob[label][word] = (count+1)/total_words

    return word_counts_prob

def classify_article(priors, word_counts_prob, vocab, tokens):
    class_probabilities = {}
    for label in priors.keys():
        log_prior = np.log(priors[label])

        for word in tokens:
            if word in vocab:
                log_prior += np.log(word_counts_prob[label].get(word, 1 / (len(vocab) + 1)))

        class_probabilities[label] = log_prior

    return max(class_probabilities, key=class_probabilities.get)

def make_predictions(priors, likelihoods, vocabulary, test_tokens_content):
    predictions_list = []
    for filename, tokens in test_tokens_content.items():
        predicted_label = classify_article(priors, likelihoods, vocabulary, tokens)
        predictions_list.append((filename, predicted_label))
    return predictions_list


def main():
    # Ask for input documents
    training_labels_file = input("Please enter the name of the file containing the list of labeled training documents: ")
    test_list_file = input("Please enter the name of the file containing the list of unlabeled test documents: ")

    base_dir = os.path.dirname(training_labels_file)
    corpus_name = os.path.basename(training_labels_file).split('_')[0]

    # Process training set
    labels, train_file_paths = read_labels(training_labels_file)
    train_files_content = read_file(base_dir, train_file_paths)
    train_tokens_content = tokenize_file(train_files_content)
    vocabulary = build_vocabulary(train_tokens_content)
    priors = calculate_prior(labels)
    likelihoods = calculate_likelihood(train_tokens_content, vocabulary, labels)

    # Process test set
    test_file_paths = read_test_list(test_list_file)
    test_files_content = read_file(base_dir, test_file_paths)
    test_tokens_content = tokenize_file(test_files_content)

    # Make predictions
    predictions_list = make_predictions(priors, likelihoods, vocabulary, test_tokens_content)
    output_file = os.path.join(base_dir, f"predicted_{corpus_name}_test.labels")
    with open(output_file, 'w') as file:
        for item in predictions_list:
            file_path = f"./{corpus_name}/test/{item[0]}"
            file.write(f"{file_path} {item[1]}\n")


if __name__ == '__main__':
    main()
