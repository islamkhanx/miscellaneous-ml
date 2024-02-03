import numpy as np
class TfidfVectorizer():
    def __init__(self):
        self.sorted_vocab = {}
        self.idf_list = {}

    def fit(self, X):

        # заполняем словарь
        all_words = set()
        [[all_words.add(word) for word in document.split(' ')]
         for document in X]
        self.sorted_vocab = {k: 0 for k in sorted(list(all_words))}

        # делаем idf для каждого слова
        self.idf_list = self.idf(X)
        # return self

    def transform(self, X):
        tf_matrix = []
        for document in X:
            tf_matrix.append(self.tf(document))
        tf_matrix = np.array(tf_matrix)
        idf_matrix = np.array(list(self.idf_list.values()))
        tfidf_matrix = tf_matrix * idf_matrix
        return tfidf_matrix

    def count_frequency(self, X):
        """
        Args:
            X (list[str]): corpus of string documents
            where tokens are seperated by whitespace

        Returns:
            list[list[int]]:  len(X) by len(|V|) matrix
            consisting of counts of each word in document
        """
        count_matrix = []
        for document in X:
            vocab_copy = self.sorted_vocab.copy()
            for word in document.split(' '):
                if word in vocab_copy:
                    vocab_copy[word] += 1
            count_matrix.append(list(vocab_copy.values()))
        return count_matrix

    def tf(self, document):
        tf_for_doc = []
        doc_tffd = document

        count_words_voc = {k: 0 for k in self.sorted_vocab.keys()}
        for word in doc_tffd.split(' '):
            if word in count_words_voc:
                count_words_voc[word] += 1

        N_doc = len(doc_tffd.split(' '))

        if N_doc == 0:
            return 0 * len(count_words_voc)
        for word in count_words_voc.keys():
            tf_for_doc.append(count_words_voc[word] / N_doc)

        return tf_for_doc

    def idf(self, docs):
        doc_count = len(docs)
        doc_freq = self.sorted_vocab.copy()
        for document in docs:
            for word in set(document.split()):
                doc_freq[word] += 1
        for word in doc_freq:
            doc_freq[word] = np.log(doc_count / doc_freq[word])
        return doc_freq



def read_input():
    n1, n2 = map(int, input().split())

    train_texts = [input().strip() for _ in range(n1)]
    test_texts = [input().strip() for _ in range(n2)]

    return train_texts, test_texts 

def solution():
    train_texts, test_texts = read_input()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_texts)
    transformed = vectorizer.transform(test_texts)

    for row in transformed:
        row_str = ' '.join(map(str, np.round(row, 3)))
        print(row_str)


if __name__ =='__main__':
    solution()
    # input:
        # 3 2
        # a a a
        # a b
        # c
        # a c
        # d
    # output
        # 0.203 0.0 0.549
        # 0.0 0.0 0.0