def test_imports():
    from vae_dist.core.CNN_regressor import CNNRegressor
    from vae_dist.core.R3CNN_regressor import R3CNNRegressor
    from vae_dist.core.R3CNN import R3CNN
    from vae_dist.core.O3VAE import R3VAE
    from vae_dist.core.VAE import baselineVAEAutoencoder
    from vae_dist.core.CNN import CNNAutoencoderLightning
    from vae_dist.core.parameters import set_enviroment
    from vae_dist.core.training_utils import construct_model
