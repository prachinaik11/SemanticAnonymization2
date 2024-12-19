# !pip install ampligraph
# !pip install tensorflow==2.12

from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.regularizers import get as get_regularizer
import numpy as np
from ampligraph.datasets import load_from_csv
from ampligraph.utils import save_model
from ampligraph.evaluation import mrr_score, hits_at_n_score
import tensorflow as tf
import ampligraph



X = {"train":[], "test":[], "valid":[], "valid1":[]}
X["train"] = load_from_csv("Original/", "x_train_ICEWS_1lac_final.tsv", sep='\t')
X["test"] = load_from_csv("Original/", "x_test_ICEWS_final.tsv", sep='\t')
X["valid"] = load_from_csv("Original/", "x_valid_ICEWS_final.tsv", sep='\t')


model = ScoringBasedEmbeddingModel(k=100,  #embedding size
                                   eta=10,
                                   scoring_type='TransE')
# Optimizer, loss and regularizer definition
optim = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss = get_loss('pairwise', {'margin': 0.5})
regularizer = get_regularizer('LP', {'p': 2, 'lambda': 1e-5})

model.compile(optimizer=optim, loss=loss, entity_relation_regularizer=regularizer)

filter = {'test' : np.concatenate((X["train"], X["valid"], X["test"]))}


# Early Stopping callback
checkpoint = tf.keras.callbacks.EarlyStopping(
    monitor='val_{}'.format('hits10'),
    min_delta=0,
    patience=5,
    verbose=1,
    mode='max',
    restore_best_weights=True
)

# Fit the model on training and validation set
model.fit(X["train"],
          batch_size=int(X["train"].shape[0] / 10),
          epochs=200,                    # Number of training epochs
          validation_freq=20,           # Epochs between successive validation
          validation_burn_in=100,       # Epoch to start validation
          validation_data=X["valid"],   # Validation data
          validation_filter=filter,     # Filter positives from validation corruptions
          callbacks=[checkpoint],       # Early stopping callback (more from tf.keras.callbacks are supported)
          verbose=True                  # Enable stdout messages
          )



# Save the model
example_name = "newModels/Ampligraph_TransE_ICEWS.pkl"
save_model(model, model_name_path=example_name)


ranks = model.evaluate(X["test"],
                       use_filter=filter,
                       corrupt_side='s,o')

# compute and print metrics:
mrr = mrr_score(ranks)
hits_10 = hits_at_n_score(ranks, n=10)
hits_1 = hits_at_n_score(ranks, n=1)
hits_5 = hits_at_n_score(ranks, n=5)
print("MRR: %f, Hits@10: %f" % (mrr, hits_10))
print("hits_1: %f, hits_5: %f" % (hits_1, hits_5))


invalid_keys = model.get_invalid_keys(X["test"])

print("Test : ")
print("Invalid keys sub:", (invalid_keys[0]))
print("Invalid keys rel:", (invalid_keys[1]))
print("Invalid keys obj:", (invalid_keys[2]))
invalid_sub_set = set(invalid_keys[0])

invalid_keys = model.get_invalid_keys(X["valid"])
print("valid : ")
print("Invalid keys sub:", (invalid_keys[0]))
print("Invalid keys rel:", (invalid_keys[1]))
print("Invalid keys obj:", (invalid_keys[2]))