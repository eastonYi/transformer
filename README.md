# A Simple Version of TensorFlow Implementation of the Transformer: <https://arxiv.org/abs/1706.03762>

This project is originally forked from <https://github.com/Kyubyong/transformer>. 
Lots of change has been made to make it easy-to-use and flexible. 
Several new features have been implemented and tested.
Some functions are taken from <https://github.com/tensorflow/tensor2tensor>.

## Highlight Features
- Batching data with bucket mechanism, which allows higher utilization of computational resources.
- Beam search that support batching and length penalty.
- Using yaml to config all hyper-parameters, as well as all other settings.
- Supporting caching decoder outputs, which accelerates decoding on CPUs.

## Contact
Raise an issue on [github](https://github.com/chqiwang/transformer) or email to <chqiwang@126.com>.
