# SPAA: Semantic-Preserving Adversarial Attack #

--------

In this project, we propose an approach to harness the SPA attack via learning to learning approach. 


-------

### Data Pro-processing

MNIST-ROT dataset generation: run `adv_defence/process_mnist_rot.py`


### Model Training

Model definition: `adv_defence\models.py`

Configuration: `adv_defence\config.py`

Model training code: `adv_defence\trainer_adv_example_2.py`

To start model training, run `main.py`

The a demo of training script is saved in `script.sh` as follows:
```bash
python main.py --dataset mnist_rot --f_pretrain True --save_step 100 \
--g_optimizer adam --g_deeper_layer True --g_base_channel_dim 8 --img_size 32 \
--adv_loss_lambda 0.1 --g_msp_lambda 0.0001 --g_vae_lambda 0.0 \
--g_rec_lambda 0.001 --comment Rece-3Adv01Mspe-4
```
