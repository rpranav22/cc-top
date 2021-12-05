from bertopic import BERTopic
import hdbscan
import pickle

topic_model = BERTopic.load("../experiments/bertopic_v1")

print(topic_model.get_topic_info())

cluster_assignment = pickle.load(open('../experiments/cluster_probs_hdbscan_bertopic.pickle', 'rb'))
print(cluster_assignment.keys(), len(cluster_assignment['docs']), len(cluster_assignment['probs']))

clustering_model = pickle.load(open('../experiments/clustering_model.pickle', 'rb'))


print(type(clustering_model))
