from ml_core.config import Config
from ml_core.models.stclassifier import PseTae_pretrained


def load_model(config: Config):
    model_config = dict(
        input_dim=config.INPUT_DIM,
        mlp1=config.MLP1,
        pooling=config.POOLING,
        mlp2=config.MLP2,
        n_head=config.N_HEAD,
        d_k=config.D_K,
        mlp3=config.MLP3,
        dropout=config.DROPOUT,
        T=config.T,
        len_max_seq=config.LMS,
        positions=None,
        mlp4=config.MLP4
    )
    
    if config.GEOMETRIC_FEATURES:
        model_config.update(with_extra=True, extra_size=4)
    else:
        model_config.update(with_extra=False, extra_size=None)

    model = PseTae_pretrained(
        config.WEIGHT_DIR,
        model_config,
        device=config.DEVICE,
        fold=config.FOLD_NUM
    )

    return model