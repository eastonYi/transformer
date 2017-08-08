# A Simple Version of TensorFlow Implementation of the Transformer: <https://arxiv.org/abs/1706.03762>

This project is originally forked from <https://github.com/Kyubyong/transformer>.
Some core code is taken from <https://github.com/tensorflow/tensor2tensor>.

## Highlight Features
- Data batching with bucket mechanism, which allows higher utilization of computational resources.
- Beam search that support batching and length penalty.
- Use yaml to config all hyper-parameters, as well as all other settings.
- Support caching decoder outputs, which accelerates decoding on CPUs.

## Contact
Raise an issue on [github](https://github.com/chqiwang/transformer) or email to <chqiwang@126.com>.
