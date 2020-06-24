import nltk
import sys
import os
import math
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


# When was Python 3.0 released?
# What are the types of supervised learning?
# How do neurons connect in a neural network?
# How did the term machine learning coined?
# What is python features and philosophy?


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))
    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    contents = dict()
    for filename in os.listdir(directory):
        path = os.path.join(directory, str(filename))

        f = open(path)
        content = f.read().replace("\n", " ")
        f.close()
        contents[filename] = content

    return contents
    


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.
    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stopwords = nltk.corpus.stopwords.words("english")
    return list(
        word.lower() for word in nltk.word_tokenize(document) if word.lower() not in stopwords and word.lower() not in string.punctuation
    )
    

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}

    for content in documents:
        for word in documents[content]:
            if word in idfs:
                continue
            else:
                count = 0
                total = 0
                for i in documents:
                    total += 1
                    if word in documents[i]:
                        count += 1
                idfs[word] = math.log(float(total/count))

    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = {}

    for file in files:
        sum = 0
        for word in query:
            #tf * idf
            sum += files[file].count(word) * idfs[word]

        tf_idfs[file] = sum

    sorted_tf_idfs = list(sorted(tf_idfs.keys(), key=lambda x : tf_idfs[x], reverse=True))

    return sorted_tf_idfs[:n]
    


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    rank = {}

    for sentence in sentences:
        sum = 0
        tf = 0
        
        count = len(sentences[sentence])
        for word in query:
            tf += sentences[sentence].count(word)
            #matching word measure
            if word in sentences[sentence]:
                sum += idfs[word]
            
        rank[sentence] = (sum, float(tf/count))

    sorted_rank = list(sorted(rank.keys(), key=lambda x: (rank[x][0], rank[x][1]), reverse=True))
    # for i in range(2):
    #     print(sorted_rank[i])
    #     print(rank[sorted_rank[i]])

    return sorted_rank[:n]

if __name__ == "__main__":
    main()


