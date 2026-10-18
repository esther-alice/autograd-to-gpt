# autograd-to-gpt

A deep learning stack built from scratch: an autograd engine, a small neural-net
library, and (in progress) a GPT-style transformer.

## Status
- [x] Scalar autograd engine (`autograd/value.py`)
- [x] Neuron / Layer / MLP (`autograd/nn.py`)
- [x] Gradient checks vs. numerical differentiation and PyTorch (`tests/`)
- [ ] Tensor autograd
- [ ] Transformer + BPE tokenizer
- [ ] Experiments and write-up

## Run
```bash
pip install -r requirements.txt
pytest
python -m experiments.train_moons
```
