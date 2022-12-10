# Serve speaker verification model using GRPC
This repo is used to serve speaker verification model that is implemented in: https://github.com/hoanduy27/PyTorch_Speaker_Verification (forked from HarryVolek's)

To start the service, run:
```
python server.py <model_path> <config_file>
```

1. `model_path`: The `.model` file after training the model
2. `config_file`: Config file in `yaml` format. Example is in `config/verifier_conf.yaml`. The parameter in this config should match with the training config.

# Example run
1. Install requirements: 
```sh
pip install -r requirements.txt
```
2. Ecport variables
```sh
export dev.env
```
3. Run the server

    3.1. Model trained with Softmax GE2E Loss
    ```sh
    python server.py model/final_epoch_10000_batch_id_4.model config/verifier_config.yaml
    ```

    3.2. Model trained with Contrastive GE2E Loss
    ```sh
    python server.py model/final_epoch_10000_batch_id_4_contrastive_loss.model config/verifier_config.yaml
    ```

4. Open another terminal and run `client.py` to test the server with sample audios in `sample_test/si`
```sh
python client.py
```