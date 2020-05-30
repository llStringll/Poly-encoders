# Poly-encoders
Poly-encoder architecture and pre-training pipeline implementation (pytorch) [Poly-encoders: architectures and pre-training
strategies for fast and accurate multi-sentence scoring](https://arxiv.org/pdf/1905.01969.pdf)

This encoder inherits [HuggingFace's BeRT](https://huggingface.co/transformers/model_doc/bert.html) class(pre-trained), and fine tunes on it, with two extra attention layers on the output, one with 'm' trainable queries(context codes, as referred in  the paper), and the other with the aggregated candidate output embedding as the query. The context encoder and the candidate encoder here, are both distillBeRTs. Output of this encoder is just softmax of the score of similarity b/w context and candidate embedding(dot product basically).

Ps- Paper mentions they pre-trained BeRT on tasks that are more similar to the downstream tasks on which they fine-tune the poly-encoder later, yielding better results (in terms of rate of convergance during fine-tuning and accuracy)
