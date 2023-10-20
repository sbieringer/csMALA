# csMALA

This Repo contains the Numerics of our Paper ...

## Structure

The Plots of the Numerics Section are generated in the Notebooks. Beyond that, <code>StochasticMH.ipynb</code>, also contains a training example using our implementation of MALA.
  * <code>src/MALA.py</code> defines our MALA implementation and can by used in exchange for your usual PyTorch Optimizer
  * <code>src/util</code> contains the risk implementations, including our corrected risk term, as well as the model definition
  * <code>src/datasets</code> contains toy data und Bernoulli-sampling Dataloader
  * <code>src/uncertimators</code> contains wrappers around  MALA and propabilistic training for easier use
