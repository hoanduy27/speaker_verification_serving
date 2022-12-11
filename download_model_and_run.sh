model_name=vivos_noise_softmax_ge2e.model
config=config/verifier_conf.yaml

mkdir -p model
cd model/
if [ ! -f ${model_name} ]; then
    gdown 1_d6yp4yLqAQJ9tg7mgU2skABiTq_vSNy
fi
cd ..
python server.py model/${model_name} ${config}
