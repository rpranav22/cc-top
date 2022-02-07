import os
import pdb
from statistics import mode
import tempfile
from typing import Any
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

from sscc.metrics import Evaluator

# embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# # Corpus with example sentences
# corpus = ['A man is eating food.',
#           'A man is eating a piece of bread.',
#           'A man is eating pasta.',
#           'The girl is carrying a baby.',
#           'The baby is carried by the woman',
#           'A man is riding a horse.',
#           'A man is riding a white horse on an enclosed ground.',
#           'A monkey is playing drums.',
#           'Someone in a gorilla costume is playing a set of drums.',
#           'A cheetah is running behind its prey.',
#           'A cheetah chases prey on across a field.'
#           ]
# corpus_embeddings = embedder.encode(corpus)
# pdb.set_trace()
# # Perform kmean clustering
# num_clusters = 5
# clustering_model = KMeans(n_clusters=num_clusters)
# clustering_model.fit(corpus_embeddings)
# cluster_assignment = clustering_model.labels_

# clustered_sentences = [[] for i in range(num_clusters)]
# for sentence_id, cluster_id in enumerate(cluster_assignment):
#     clustered_sentences[cluster_id].append(corpus[sentence_id])

# for i, cluster in enumerate(clustered_sentences):
#     print("Cluster ", i+1)
#     print(cluster)
#     print("")



class BERTKmeans():

    def __init__(self, model, loss, num_classes, **kwargs) -> None:
        self.model = model
        self.num_classes = num_classes
        self.clustering_model = KMeans(n_clusters=self.num_classes)
        # self.base_model = kwargs['architecture']['base_model']
        self.embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')


    def fit_model(self, X_train):
        corpus_embeddings = self.embedder.encode(X_train)

        self.clustering_model.fit(corpus_embeddings)

        cluster_assignment = self.clustering_model.labels_
        return cluster_assignment
       


    def evaluate(self, labels: Any, batch: Any, confusion: Any=Evaluator(20), part: str='val', current_epoch: int=0, logger: Any=None, true_k: int=20):
        """Evaluate model on any dataloader during training and testings

        Args:
            batch (Any): data loader for evaluation
            confusion (Any, optional): the evaluator object as definsed in sscc.metrics. Defaults to Evaluator(10).
            part (str, optional): train/test/val . Defaults to 'val'.
            logger (Any, optional): any pl logger. Defaults to None.
            true_k (int, optional): true clusters, important for confusion matrix plotting. Defaults to 10.

        Returns:
            None
        """   
        if part == 'test': y, pred = [], [] 
        

        confusion.add(batch, labels)
        if part == 'test':
            y.extend(labels.cpu().detach().numpy())
            pred.append(batch.cpu().detach().numpy())

        if part == 'test': 
            pred = np.concatenate(pred, axis=0)


        confusion.optimal_assignment(confusion.k)
        print('\n')
        print(f"{part} ==> clustering scores: {confusion.clusterscores()}")
        print(f"{part} ==> accuracy: {confusion.acc()}")
        # store in dictionary for mlflow logging
        eval_results = {f'{part}_acc': confusion.acc()}
        eval_results.update({f'{part}_{key}': value for key, value in confusion.clusterscores().items()})

        if part != 'train':
            epoch_string=f'{current_epoch}'.zfill(4)
            logger.experiment.log_figure(figure=confusion.plot_confmat(title=f'{part} set, epoch {current_epoch}',
                                                                       true_k=true_k),
                                         artifact_file=f'confmat_{part}_{epoch_string}.png',
                                         run_id=logger.run_id)
        if part == 'test':
            # log the test predictions
            # pdb.set_trace()
            # final_results = {f'yhat_p_{cl}': pred[:, cl] for cl in range(pred.shape[1])}
            final_results = {}
            final_results['y'] = y
            final_results['yhat'] = np.argmax(pred, 0)
            final_results = pd.DataFrame(final_results)
            with tempfile.TemporaryDirectory() as tmp_dir:
                storage_path = os.path.join(tmp_dir, 'test_preds.csv')
                final_results.to_csv(storage_path)
                logger.experiment.log_artifact(local_path=storage_path, run_id=logger.run_id)

        return eval_results