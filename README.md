# Serve speaker verification model using GRPC
This repo is used to serve speaker verification model that is implemented in: https://github.com/hoanduy27/PyTorch_Speaker_Verification (forked from HarryVolek's)


## 1. Requirements
1. `ffmpeg`: [click here](https://ffmpeg.org/download.html)
2. python >= 3.8
3. Packages:
```
grpcio==1.51.1
grpcio-tools==1.51.1
protobuf==4.21.10
pydub==0.25.1
scipy==1.9.3
soundfile==0.11.0
librosa==0.9.2
torch==1.13.0
numpy==1.23.5
pyyaml==6.0
```


## 2. How to run
1. Export variables:
```sh
export dev.env
```

2. Start the server
```
python server.py <model_path> <config_file>
```

- `model_path`: The `.model` file after training the model
- `config_file`: Config file in `yaml` format. Example is in `config/verifier_conf.yaml`. The parameter in this config should match with the training config.
- Please refer to [Pretrained models](#pretrained-models).

3. [OPTIONAL] Open another terminal and run `client.py` to test the server with sample audios in `sample_test/si`
```sh
python client.py
```

## 3. Pretrained models <a name="pretrained-model"></a>
Pretrained model can be found [here](https://drive.google.com/drive/folders/1s5UfdLmR2yocsPKUStXSYGNsrvTqgIAW?usp=share_link). 

All the models use the same config in [config/verfier_conf.yaml](config/verifier_conf.yaml)

|model_name|train data|loss|
|:---------:|:---------:|:---------:|
|vivos_clean_softmax_ge2e.model|vivos (clean)|Softmax GE2E|
|vivos_clean_contrastive_ge2e.model|vivos (clean)|Contrastive GE2E|
|vivos_noise_softmax_ge2e.model|vivos (clean + noise)|Softmax GE2E|
|vivos_noise_contrastive_ge2e.model|vivos (clean + noise)|Contrastive GE2E|