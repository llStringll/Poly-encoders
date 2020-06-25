# Poly-encoders
Poly-encoder architecture and pre-training pipeline implementation (pytorch) [Poly-encoders: architectures and pre-training
strategies for fast and accurate multi-sentence scoring](https://arxiv.org/pdf/1905.01969.pdf)

This encoder inherits [HuggingFace's BeRT](https://huggingface.co/transformers/model_doc/bert.html) class(pre-trained), and fine tunes on it, with two extra attention layers on the output, one with 'm' trainable queries(context codes, as referred in  the paper), and the other with the aggregated candidate output embedding as the query. The context encoder and the candidate encoder here, can be either distillBeRT or BeRT. Output of this encoder is just log-softmax of the score of similarity b/w context and candidate embedding log(softmax((dot product))). This score drives towards 0 for correct response and towards -inf for negative candidates(picked from candidate responses of others entries in the batch). All repsonses in a batch attend to all context vectors of that batch, the ones with label==1 are considered positive candidates, rest are considered negative candidates, hence allowing re-use of the computed candidate embeddings for all context vectors in a given batch.

During training, the encoder.py returns masked log-softmax of the dot product b/w candidate and transformed context vectors. The mask, zeros out the log-softmax of negativr candidates(which should be -inf in ideal case), so output of this would be log-softmax of positive candidates for every context in the batch.

During eval, encoder.py returns dot product itself, w/o applying any logisitc fncn(like log softmax during training), and the argmax of this dot product is comapred to argmax of labels to measure eval accuracy %

Paper mentions they pre-trained BeRT on tasks that are more similar to the downstream tasks on which they fine-tune the poly-encoder later, yielding better results (in terms of rate of convergance during fine-tuning and accuracy)

As default
- Data path - ./data/
- Pretrained BeRT path - ./ckpt/pretrained/
- Max content length, 100
- Max response length, 50
- Batch size, 32
- Number of m-query codes for poly-encoder, 16
- Adam initial eta, 5e-05, warmup steps, 2000
- Epochs, 3

Can be changed via args to train.py


- Results mentioned are on Ubuntu Dialogue Corpus V2.0 (https://arxiv.org/abs/1506.08909), downloaded from (http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ubuntu_dialogs.tgz).
- With BeRT-small-uncased pretrained (from Hugging-Face pretrained models, this repo uses bert base uncased. Or get one from https://github.com/google-research/bert)


*The project is not complete yet!*
