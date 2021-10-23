# Looking for Confirmations: An Effective and Human-Like Visual Dialogue Strategy (EMNLP 2021)

Pre-processed dataset, visual features, and pre-trained models can be found here: https://drive.google.com/drive/folders/1zfkSRuqV_yQHR-1HnSydNub4OOH5wRmN?usp=sharing . Place all the files in the *data* directory.

You can find the list of requirements in *confirm_it_req.txt*

This code is based on Shekhar et al. (2019) code released at: https://github.com/shekharRavi/Beyond-Task-Success-NAACL2019

## Training
To run the Confirm-it training script, use the following command:
 ```
CUDA_VISIBLE_DEVICES=0 python -m train.SL.train_confirm_it_3tasks -modulo 7 -no_decider -exp_name confirm_it -bin_name confirm_it
```

## Inference
Run *inference_confirm_it.py*.

*Detailed instructions coming soon*

## Reference

```
@inproceedings{testoni2021confirm_it,
  title = {Looking for Confirmations: An Effective and Human-Like Visual Dialogue Strategy},
  author = {Alberto Testoni and Raffaella Bernardi},
  booktitle = {Proceedings of The 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021},
  year = {2021}
}
```
