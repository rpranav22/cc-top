import gensim
# import nltk
from nltk.corpus import stopwords
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy

class LDA():
    def __init__(self, num_topics: int = 20, remove_stopwords: bool = True) -> None:
        self.num_topics = num_topics
        self.remove_stopwords = remove_stopwords
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
        corpus = [self.id2word.doc2bow(text) for text in self.texts]

        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=self.id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        
        print(self.lda_model.print_topics())

    




    
    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    def remove_all_stopwords(self, texts):
        return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in self.stopwords] for doc in texts]

    def make_bigrams(self, texts):
        return [self.bigram_mod[doc] for doc in texts]

    def lemmatization(self, texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    def evaluate(self):

        coherence_model_lda = CoherenceModel(model=self.lda_model, texts=self.texts, dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)