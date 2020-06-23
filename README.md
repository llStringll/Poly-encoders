# Poly-encoders
Poly-encoder architecture and pre-training pipeline implementation (pytorch) [Poly-encoders: architectures and pre-training
strategies for fast and accurate multi-sentence scoring](https://arxiv.org/pdf/1905.01969.pdf)

This encoder inherits [HuggingFace's BeRT](https://huggingface.co/transformers/model_doc/bert.html) class(pre-trained), and fine tunes on it, with two extra attention layers on the output, one with 'm' trainable queries(context codes, as referred in  the paper), and the other with the aggregated candidate output embedding as the query. The context encoder and the candidate encoder here, are both distillBeRTs. Output of this encoder is just softmax of the score of similarity b/w context and candidate embedding(dot product basically). This score drives towards 1 for correct response and towards 0 (via softmax) for negative candidates(picked from candidate responses of others entries in the batch)

Paper mentions they pre-trained BeRT on tasks that are more similar to the downstream tasks on which they fine-tune the poly-encoder later, yielding better results (in terms of rate of convergance during fine-tuning and accuracy)

As default
- Data path - ./data/
- Pretrained BeRT path - ./ckpt/pretrained/
- Max content length, 128
- Max response length, 64
- Batch size, 32
- Number of m-query codes for poly-encoder, 16
- Adam initial eta, 5e-05, warmup steps, 2000
- Epochs, 3
Can be changed via args to train.py

Results mentioned are on Ubuntu Dialogue Corpus V2.0 (https://arxiv.org/abs/1506.08909), downloaded from (http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ubuntu_dialogs.tgz). With BeRT-small-uncased pretrained (https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8.zip)


*The project is not complete yet!*
