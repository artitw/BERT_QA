# BERT-QA
Build question-answering systems using state-of-the-art pre-trained contextualized language models, e.g. BERT. We are working to accelerate the development of question-answering systems based on BERT and TF 2.0!

## Background

This project is based on our study: [Question Generation by Transformers](https://arxiv.org/abs/1909.05017).

### Citation

To cite this work, use the following BibTeX citation.

```
@article{question-generation-transformers@2019,
  title={Question Generation by Transformers},
  author={Kriangchaivech, Kettip and Wangperawong, Artit},
  journal={arXiv preprint arXiv:1909.05017},
  year={2019}
}
```

## Requirements
TensorFlow 2.0 will be installed if not already on your system

## Installation
```
pip install bert_qa
```

## Example usage
Run Colab demo notebook [here](https://colab.research.google.com/drive/1-tLvxSuI0ik2BaruaY_Ivoh_4eobWzEW).

### download pre-trained models and SQuAD data
```
wget -q https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/cased_L-12_H-768_A-12.tar.gz
tar -xvzf cased_L-12_H-768_A-12.tar.gz
mv -f home/hongkuny/public/pretrained_models/keras_bert/cased_L-12_H-768_A-12 .
```

### download SQuAD data
```
wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
```

### import, initialize, pre-process data, finetune, and predict!
```
from bert_qa import squad
qa = squad.SQuAD()
qa.preprocess_training_data()
qa.fit()
predictions = qa.predict()
```

### evaluate
```
import json
json_data = open('dev-v1.1.json')
data = json.load(json_data)
qa.evaluate(data, predictions)
```

## Advanced usage

### Model type
The default model is an uncased Bidirectional Encoder Representations from Transformers (BERT) consisting of 12 transformer layers, 12 self-attention heads per layer, and a hidden size of 768. Below are all models currently supported that you can specify with `hub_module_handle`. We expect that more will be added in the future. For more information, see [TensorFlow's BERT GitHub](https://github.com/tensorflow/models/blob/master/official/nlp/bert/README.md).

*   **[`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/wwm_uncased_L-24_H-1024_A-16.tar.gz)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Large, Cased (Whole Word Masking)`](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/wwm_cased_L-24_H-1024_A-16.tar.gz)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Base, Uncased`](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12.tar.gz)**:
    12-layer, 768-hidden, 12-heads, 110M parameters
*   **[`BERT-Large, Uncased`](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/uncased_L-24_H-1024_A-16.tar.gz)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters
*   **[`BERT-Base, Cased`](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/cased_L-12_H-768_A-12.tar.gz)**:
    12-layer, 768-hidden, 12-heads , 110M parameters
*   **[`BERT-Large, Cased`](https://storage.googleapis.com/cloud-tpu-checkpoints/bert/keras_bert/cased_L-24_H-1024_A-16.tar.gz)**:
    24-layer, 1024-hidden, 16-heads, 340M parameters


## Contributing
BERT-QA is an open-source project founded and maintained to better serve the machine learning and data science community. Please feel free to submit pull requests to contribute to the project. By participating, you are expected to adhere to BERT-QA's [code of conduct](CODE_OF_CONDUCT.md).

## Questions?
For questions or help using BERT-QA, please submit a GitHub issue.
