from nltk.tokenize import word_tokenize
from string import punctuation
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

factory = StopWordRemoverFactory()
stop_word = factory.get_stop_words()
junks = stop_word + list(punctuation) + ["``", "''", "--"]


def safe_division(up, bottom):
    if up == 0:
        return 0
    return up / bottom


class segment_evaluation:
    def __init__(self, labels, classes):
        self.labels = labels
        self.classes = classes
        self.precision, self.recall, self.f_measure = self.get_value()

    def get_value(self):
        k = max(self.labels + 1)
        index = np.arange(self.labels.shape[0])

        E = np.zeros((3, k), dtype=float)
        for i in range(k):
            true_label_i = index[self.labels == i]
            true_class_i = index[self.classes == i]

            TP = np.intersect1d(true_label_i, true_class_i).shape[0]

            E[0][i] = safe_division(TP, true_class_i.shape[0])
            E[1][i] = safe_division(TP, true_label_i.shape[0])
            E[2][i] = safe_division(2 * E[0][i] * E[1][i], E[0][i] + E[1][i])

        E = np.round((E.sum(1) / k) * 100, 3)
        return E


def plot_image(titles, matrix, subplot=1, line=False, segments=None):

    if subplot == 1:
        plt.title(titles)
        plt.imshow(matrix, cmap='Greys_r')
    else:
        fig, axs = plt.subplots(1, subplot)
        for title, ax, m in zip(titles, axs, matrix):
            ax.set_title(title)
            ax.imshow(m, cmap='Greys_r')

    if line:
        for segment in segments[:-1]:
            plt.axvline(x=segment[-1], color='r')
            plt.axhline(y=segment[-1], color='r')
    plt.show()


def plot_density(range_k, density):
    plt.plot(range_k, density)
    plt.xlabel('K')
    plt.ylabel('density')
    plt.show()


class preparing_data:

    def __init__(self, file, index_surat=2, normalize_tf=True, quran=True):
        self.file = file
        self.index = str(index_surat)
        if quran:
            self.raw, self.label = self.read_raw_data_quran()
        else:
            self.raw, self.label = self.read_raw_data_article()
        self.clean = self.pre_processing()
        self.vocab = self.get_vocabulary()
        self.tf, self.tfidf = self.weighting(normalize_tf=normalize_tf)

    def read_raw_data_quran(self):
        with open(self.file) as f:
            reader = f.read().splitlines()
        raw = [i.split('|')[-1] for i in reader if i.startswith(self.index + '|')]
        with open('pokok bahasan new.txt') as f:
            reader = f.read().splitlines()
        nums = '0'*(3 - len(str(self.index))) + str(self.index)
        reader = [i.split('|')[-1] for i in reader if i.startswith(nums)]
        label = []
        for i, line in enumerate(reader):
            a, b = [int(i) for i in line.split()[0].split(':')[-1].split('-')]
            label.append([i] * (b - a + 1))
        return raw, np.concatenate(label)

    def read_raw_data_article(self):
        ranges = range(80, 100)
        with open(self.file) as f:
            reader = f.read().splitlines()
        raw = [i.split(' >> ')[-1] for i in reader if i.startswith(tuple(str(j) + ' ' for j in ranges))]
        label = [int(i.split(' >> ')[0]) for i in reader if i.startswith(tuple(str(j) + ' ' for j in ranges))]
        return raw, np.array(label) - min(label)

    def pre_processing(self):
        clean = []
        for doc in self.raw:
            doc = word_tokenize(doc.lower())  # case folding & tokenizing
            temp = []
            for word in doc:
                if word in junks:  # remove stop words
                    continue
                temp.append(word)
            clean.append(temp)
        return clean

    def get_vocabulary(self):
        return sorted(set(np.concatenate(self.clean)))

    def weighting(self, normalize_tf=True):
        m = len(self.clean)
        n = len(self.vocab)

        tf = np.zeros((m, n), dtype=float)
        for i, docs in enumerate(self.clean):
            for j, word in enumerate(self.vocab):
                if normalize_tf:
                    tf[i, j] = docs.count(word) / len(docs)
                else:
                    tf[i, j] = docs.count(word)
        idf = np.log10(m / np.where(tf > 0, 1, 0).sum(0))
        tfidf = np.array([tf[i] * idf for i in range(m)])
        return tf, tfidf


class lsa_segmentation:

    def __init__(self, tf, weight, k=10, local_region=5):
        self.k = k
        self.tf = tf
        self.weight = weight
        self.delta = self.singular_value_decomposition()
        self.delta_k = self.delta[:, :k]
        self.M = self.cosine_similarity()
        self.R = self.ranking(r=local_region)
        self.cluster, self.density, self.label = self.divisive_clustering()

    def singular_value_decomposition(self):
        delta, _, _ = np.linalg.svd(self.weight.T)
        return delta

    def cosine_similarity(self):
        lamb = np.dot(self.tf, self.delta_k)
        cosine = 1 - pairwise_distances(lamb, metric='cosine')
        return cosine

    def ranking(self, r=5):
        m = self.M.shape[0]
        rank = np.zeros(self.M.shape, dtype=float)
        for i in range(m):
            y1 = i + r + 1
            x1 = 0 if i - r < 0 else i - r
            for j in range(m):
                y2 = j + r + 1
                x2 = 0 if j - r < 0 else j - r
                local = self.M[x1: y1][:, x2: y2]
                lower = np.where(local < self.M[i, j], 1, 0)
                rank[i, j] = lower.sum() / (lower.size - 1)
        return rank

    def split_segment(self, index):
        mask = np.copy(self.R)[index][:, index]
        m = mask.shape[0]
        myu = np.zeros(m - 1, dtype=float)
        for i in range(m - 1):
            seg1 = mask[:i + 1][:, :i + 1]
            seg2 = mask[i + 1:][:, i + 1:]
            myu[i] = (seg1.sum() + seg2.sum()) / (seg1.size + seg2.size)
        return [list(i) for i in np.split(index, [np.argmax(myu) + 1])]

    def get_density(self, segments):
        alpha, beta = [], []
        for segment in segments:
            beta.append(self.R[segment][:, segment].sum())
            alpha.append(self.R[segment][:, segment].size)
        return sum(beta) / sum(alpha)

    def divisive_clustering(self):
        m = self.R.shape[0]
        segments = [list(range(m))]

        best_dens = 0
        while len(segments) < self.k:
            sub_index = []
            best_dens = []
            for i, segment in enumerate(segments):
                if len(segment) == 1:
                    continue
                sub_index1, sub_index2 = self.split_segment(segment)
                temp = [list(i) for i in np.copy(segments)]
                temp.remove(segment)
                temp += [sub_index1, sub_index2]
                sub_index.append(temp)
                best_dens.append(self.get_density(temp))
            index_max = best_dens.index(max(best_dens))
            segments = sub_index[index_max]
        segments = sorted(segments)
        label = np.concatenate([[i] * len(j) for i, j in enumerate(segments)])
        return segments, max(best_dens), label


def data_frame(frame, cols=None, rows=None):
    m = len(frame[0])
    n = len(frame)

    cols = [''] * m if cols is None else ['{}{}'.format(cols, i+1) for i in range(m)]
    rows = ['   '] * n if rows is None else ['   {}{}'.format(rows, i+1) for i in range(n)]

    df = pd.DataFrame(frame)
    df.columns = cols
    df.index = rows
    print(df, '\n')


var_K = False
data = preparing_data('concat.txt', index_surat=4, normalize_tf=True, quran=False)

if not var_K:
    K = max(data.label) + 1

    print('>> DATA MENTAH (1-10):')
    for idx, d in enumerate(data.raw[:10]):
        print('   dokumen{}: {}'.format(idx+1, d))
    print('')

    print('>> DATA SETELAH PREPROCESSING (1-10):')
    for idx, d in enumerate(data.clean[:10]):
        print('   dok{}: {}'.format(idx + 1, d))
    print('')

    print('>> VOCABULARY (1-99):')
    data_frame(np.array(data.vocab[:99]).reshape(33, 3))

    print('>> TERM FREQUENCY:')
    data_frame(data.tf, 'term', 'dok')

    print('>> TERM FREQUENCY - INVERSE DOCUMENT FREQUENCY:')
    data_frame(data.tfidf, 'term', 'dok')

    lsa = lsa_segmentation(data.tf, data.tfidf, k=K, local_region=5)

    print('>> SINGULAR VALUE DECOMPOSITION (U):')
    data_frame(lsa.delta, 'feature', 'term')

    print('>> K-DIMENSIONAL FEATURE:')
    data_frame(lsa.delta_k, 'feature', 'term')

    print('>> COSINE SIMILARITY ANTAR DOKUMEN:')
    cosine_sim = lsa.M
    data_frame(cosine_sim, 'dok', 'dok')

    print('>> RANK SIMILARITY:')
    ranks = lsa.R
    data_frame(ranks, 'dok', 'dok')

    image_titles = ['Similarity Matrix', 'Rank Matrix', 'Rank Matrix Segmented']
    cluster = lsa.cluster

    plot_image(image_titles[0], cosine_sim, subplot=1)
    plot_image(image_titles[1], ranks, subplot=1)
    plot_image(image_titles[2], ranks, subplot=1, line=True, segments=cluster)
    plot_image(image_titles, [cosine_sim, ranks, ranks], subplot=3, line=True, segments=cluster)

    for topic in range(K):
        print('>> TOPIC SEGMENT {}:'.format(topic+1))
        for idx in cluster[topic]:
            print('   Dok {}: {}'.format(idx+1, data.raw[idx]))
        print('')

    evaluation = segment_evaluation(data.label, lsa.label)
    print('PRECISION: {}%'.format(evaluation.precision))
    print('RECALL   : {}%'.format(evaluation.recall))
    print('F-MEASURE: {}%'.format(evaluation.f_measure))
else:
    range_K = range(2, 51)

    densities = []
    for K in range_K:
        print('Processing K = {}'.format(K))
        lsa = lsa_segmentation(data.tf, data.tfidf, k=K, local_region=5)
        densities.append(lsa.density)

    plot_density(range_K, densities)
