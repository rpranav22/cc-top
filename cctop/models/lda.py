import os
import gensim
# import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy

class LDA():
    def __init__(self,mallet=False, num_topics: int = 20, remove_stopwords: bool = True) -> None:
        self.num_topics = num_topics
        self.remove_stopwords = remove_stopwords
        self.mallet_path = 'mallet-2.0.8/bin/mallet'
        self.mallet = mallet
        if self.remove_stopwords:
            self.stopwords = stopwords.words('english')
            self.stopwords.extend(['from', 'subject', 're', 'edu', 'use'])
    
    def run_lda(self, x_train):
        
        data_words = list(self.sent_to_words(x_train))
        self.bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)        
        self.bigram_mod = gensim.models.phrases.Phraser(self.bigram)

        # Remove Stop Words
        data_words_nostops = self.remove_all_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = self.make_bigrams(data_words_nostops)

        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(data_words_bigrams,nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        # Create Dictionary
        self.id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        self.texts = data_lemmatized

        print("lemmatisation done, starting lda")

        # Term Document Frequency
        self.corpus = [self.id2word.doc2bow(text) for text in self.texts]

        candidate_models, coherence_values = self.compute_coherence_values()

        self.plot_coherence_scores(coherence_values)
    



    
    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    def remove_all_stopwords(self, texts):
        return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in self.stopwords] for doc in texts]

    def make_bigrams(self, texts):
        return [self.bigram_mod[doc] for doc in texts]

    def lemmatization(self, texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    
    def plot_coherence_scores(self,coherence_values, limit=30, start=2, step=3):
        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

        for m, cv in zip(x, coherence_values):
            print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

    def compute_coherence_values(self, limit=30, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            if self.mallet:
                print("mallet")
                candidate_model = gensim.models.wrappers.LdaMallet(self.mallet_path, corpus=self.corpus, num_topics=num_topics, id2word=self.id2word)
            else:
                candidate_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                            id2word=self.id2word,
                                                            num_topics=num_topics, 
                                                            random_state=100,
                                                            update_every=1,
                                                            chunksize=100,
                                                            passes=10,
                                                            alpha='auto',
                                                            per_word_topics=True)
            model_list.append(candidate_model)
            coherence_values.append(self.evaluate(candidate_model))

        return model_list, coherence_values
    
    def evaluate(self, candidate_model):
        # if not os.path.exists(self.dataset_path):
        #     os.makedirs(self.dataset_path)
        coherence_model_lda = CoherenceModel(model=candidate_model, texts=self.texts, dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        return coherence_lda