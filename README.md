# sentiment-analysis
Fine-Tuning RoBERTa for Graph-Based Sentiment Analysis

The finetuned model can be found on the huggingface hub

https://huggingface.co/aymie-oh/roberta-emotion-classification


**Install dependencies:**
- pip install pandas scikit-learn
- pip install transformers[torch]

**To utilize a local GPU:**
- pip install torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 (I use CUDA 12.8 but pytorch offers additional versions)

**for Pytorch GPU version setup please refer to the documentation**
https://pytorch.org/get-started/locally/

The dataset utllized for this project is a reannotated version of the GoEmotions dataset curated by Google

@inproceedings{demszky2020goemotions,
 author = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
 booktitle = {58th Annual Meeting of the Association for Computational Linguistics (ACL)},
 title = {{GoEmotions: A Dataset of Fine-Grained Emotions}},
 year = {2020}
}

