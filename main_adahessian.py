import torch
from allennlp.common import Params
from allennlp.data import Vocabulary, PyTorchDataLoader
from allennlp.data.dataset_readers import Conll2003DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.modules import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import TokenCharactersEncoder
from allennlp.training.learning_rate_schedulers import ReduceOnPlateauLearningRateScheduler
from allennlp_models.tagging import CrfTagger
from torch.nn import LSTM

from optimizers.adahessian import Adahessian
from broka_trainer import AdaTrainer
from optimizers.ranger_optimizer import Ranger


def get_model(vocab: Vocabulary) -> CrfTagger:
    hidden_dimension = 256
    layers = 2
    bidirectional = True
    total_embedding_dim = 0

    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=100, trainable=True
    )

    total_embedding_dim += 100

    params = Params(
        {
            "embedding": {"embedding_dim": 16, "vocab_namespace": "token_characters"},
            "encoder": {
                "type": "cnn",
                "embedding_dim": 16,
                "num_filters": 128,
                "ngram_filter_sizes": [3],
                "conv_layer_activation": "relu",
            },
        }
    )
    char_embedding = TokenCharactersEncoder.from_params(vocab=vocab, params=params)
    total_embedding_dim += 128

    active_embedders = {
        "tokens": token_embedding,
        "token_characters": char_embedding,
    }

    word_embeddings = BasicTextFieldEmbedder(active_embedders)

    network = LSTM(
        total_embedding_dim,
        hidden_dimension,
        num_layers=layers,
        batch_first=True,
        bidirectional=bidirectional
    )

    encoder = PytorchSeq2SeqWrapper(network, stateful=True)

    # Finally, we can instantiate the model.
    model = CrfTagger(
        vocab=vocab,
        text_field_embedder=word_embeddings,
        encoder=encoder,
        label_encoding="BIO",
        constrain_crf_decoding=True,
        calculate_span_f1=True,
    )
    return model


def train(train, validation, optimizer_name):
    batch_size = 32
    learning_rate = 0.01
    max_iterations = 100

    token_indexer = {
        "tokens": SingleIdTokenIndexer(),
        "token_characters": TokenCharactersIndexer(min_padding_length=3),
    }

    reader = Conll2003DatasetReader(token_indexer)

    train_dataset = reader.read(train)

    validation_dataset = reader.read(validation)

    # Once we've read in the datasets, we use them to create our <code>Vocabulary</code>
    # (that is, the mapping[s] from tokens / labels to ids).
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

    # Set variables

    model = get_model(vocab)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    if optimizer_name == 'adahessian':
        optimizer = Adahessian(
            model.parameters(), lr=learning_rate, block_length=2
        )
    elif optimizer_name == 'ranger':
        optimizer = Ranger(model.parameters(), lr=learning_rate)
    else:
        raise AttributeError()

    train_dataset.index_with(vocab)
    validation_dataset.index_with(vocab)

    scheduler = ReduceOnPlateauLearningRateScheduler(
        optimizer, factor=0.5, patience=4, mode="min", verbose=True
    )

    dl = PyTorchDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    dl_validation = PyTorchDataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    trainer_model = AdaTrainer

    trainer = trainer_model(
        model=model,
        optimizer=optimizer,
        # iterator=iterator,
        grad_norm=10.0,
        data_loader=dl,
        validation_data_loader=dl_validation,
        learning_rate_scheduler=scheduler,
        patience=8,
        num_epochs=max_iterations,
        cuda_device=cuda_device,
    )
    train_metrics = trainer.train()
    print(train_metrics)


if __name__ == "__main__":

    print("Training using Adahessian")
    train(train="data/eng.testa", validation="data/eng.testa", optimizer_name="adahessian")

    print("Training using Ranger")
    train(train="data/eng.testa", validation="data/eng.testa", optimizer_name="ranger")