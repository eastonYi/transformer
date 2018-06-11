# A Simpler Implementation of the Transformer: <https://arxiv.org/abs/1706.03762>

This project is originally forked from <https://github.com/Kyubyong/transformer>. 
Lots of change has been made to make it easy-to-use and flexible. 
Several new features have been implemented and tested.
Some functions are taken from <https://github.com/tensorflow/tensor2tensor>.

## Highlight Features
- Batching data with bucket mechanism, which allows higher utilization of computational resources.
- Beam search that support batching and length penalty.
- Using yaml to config all hyper-parameters, as well as all other settings.
- Supporting caching decoder outputs, which accelerates decoding on CPUs.

## Usage
Create a new config file.

`cp config_template.yaml your_config.yaml`

Configure *train.src_path*, *train.dst_path*, *scr_vocab* and *dst_vocab* in *your_config.yaml*.
After that, run the following command to build the vocabulary files.

`python vocab.py -c your_config.yaml`
 
Edit *src\_vocab_size* and *dst\_vocab_size* in *your_config.yaml* according to the vocabulary files generated in previous step.

Run the following command to start training loops:

`python train.py -c your_config.yaml`


## Contact
Raise an issue on [github](https://github.com/chqiwang/transformer) or email to <chqiwang@126.com>.
