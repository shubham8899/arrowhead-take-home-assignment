
# Arrowhead take home assignemnt

Fine tuning transformer based NLP model (BART pretrained on CNN DailyMail) for text summarization of BBC News data. The Google Colab implementation [(here)](https://colab.research.google.com/drive/1XZeFFSA1GUpeOeBEQ1F0bO6Z2N0ZA_0a?usp=sharing) is the easiest to run. [This](https://colab.research.google.com/drive/1d4Dt81ZRuZe9l_mho0w4DaZ6vvJSz4t7?usp=sharing) separate notebook is for inference only. 

## Dataset specification
- ~2225 News Article (Title included) and Extractive Summary pairs. 1 pair had to be excluded due to utf-8 encoding issues.
- Split across 5 categories (business, entertainment, politics, sport, tech) not used for this assignment.
- 75% Train and 25% Test split data. Random state is locked and all inference/generation is on a fixed test set.

## Model specification
- BART model pre-trained on English language, and fine-tuned on CNN Daily Mail. It was introduced in the paper BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension by Lewis et al. and first released in [this repository](https://github.com/pytorch/fairseq/tree/master/examples/bart).
- BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder. BART is pre-trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text.
- BART is particularly effective when fine-tuned for text generation (e.g. summarization, translation) but also works well for comprehension tasks (e.g. text classification, question answering). This particular checkpoint has been fine-tuned on CNN Daily Mail, a large collection of text-summary pairs.
- A combination of pretraining on summarization and news data makes this ideal for this assignment

## Data preprocessing
- The data is split in category folder within articles and summary folders.
- Created a CSV combining all articles and summaries into two columns.
- No string preprocessing required for tokenization using the model tokenizer.

## Baseline
- Inference in performed on the test set using the baseline 'facebook/bart-large-cnn' model to establish baseline without finetuning.
- Four metrics: ROUGE1, ROUGE2, ROUGEL AND ROUGELSUM are calculated to assess baseline performance (state of the art evaluation metrics for text summarization).
- Considered BLEU as an additional metric, but research literature mainly uses only ROUGE (and its versions) for summarization evaluation.

## Finetuning
- Feature extraction, hyperparameter specification and training is performed to fine tune the baseline model on the train set. 
- Two versions with 1 epoch and 4 epochs are saved and evaluated.
- No more epochs are run to avoid overfitting (train loss is significantly larger than validation loss).
- Trained weight files can be found in this [Google Drive Folder](https://drive.google.com/drive/folders/1OwEki57MnUG3wyXL7Y5ZxmY-JAzSBTKP?usp=sharing) and can be added to colab.

## Evaluation
- Evaluation of the finetuned version is done in a similar manner to the baseline model.
- Results are added to the results CSV.

## Results
|          Model          |    Rouge1    |    Rouge2    | RougeL       |   RougeLSum  |
|:-----------------------:|:------------:|:------------:|--------------|:------------:|
|    **bart_baseline**    | 0.3435677847 | 0.2380172272 | 0.2576727384 | 0.2574992535 |
| **bart_finetuned_1_ep** | 0.7251790845 | 0.6333551903 | 0.5040577712 | 0.5038004591 |
| **bart_finetuned_4_ep** | 0.7713235914 | 0.6922320647 | 0.5582397274 | 0.5588417849 |               


## Run Locally
Recommend running on Google Colab using links above. Running these commands would help run this model locally.

Clone the project

```bash
  git clone https://github.com/shubham8899/arrowhead-take-home-assignment.git
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the Jupyter Notebook


## Acknowledgements
 - [Summarization Tutorial by Huggingface](https://huggingface.co/docs/transformers/tasks/summarization)
 - [Introduction to Text Summarization with ROUGE Scores](https://towardsdatascience.com/introduction-to-text-summarization-with-rouge-scores-84140c64b471)
 - [BART Research Paper](https://arxiv.org/pdf/1910.13461.pdf)
 - [BART Documentation](https://huggingface.co/docs/transformers/model_doc/bart)


## Improvements / Roadmap
- Reduce inference time by optimizing embedding size.
- Try more hyperparamters and training settings.
- Modularize code and optimize to run on most machines.
- Experiment with other similar models.
- Formatting issues with labels need to be fixed

