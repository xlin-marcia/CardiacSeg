import ml_collections

def get_example_config():
    config = ml_collections.ConfigDict()
    config.hidden_size = 384
    config.n_classes = 26
    config.transformer = ml_collections.ConfigDict()
    config.transformer.num_heads = 6
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    return config