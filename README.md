# Description
FRONTIER is a deep reinforcement learning model for portfolio management that takes investor preferences into account. The original version allows long-only trades (see [this paper](https://doi.org/10.36227/techrxiv.19165745.v1) for more details). This repo will extend the original version to allow short trades as well.

# How to use
1. Clone this repo onto your machine
2. Open development environment using one of the following methods:
    * Open in VS code using the `Remote-Containers` extension
    * Manually reproduce the development environment using the Dockerfile in `.devcontainer/`
    * Install the requiremetns in your local environment by running `pip install -r requirements.txt`
3. Browse through `code/REINFORCE/REINFORCE_soft.py` to see the details of the environment and model architecture, training, backtesting, etc.
4. Run `code/REINFORCE/train_template.ipynb` to see an example of how the models are trained
5. Run `code/REINFORCE/backtest_template.ipynb` to see an example of how a trained model is backtested

# Roadmap (TODO)
### Setting up contributors and environment to checking that everythin works as it did before
- [x] Add code from original study
- [x] Add data folders containing market data
- [x] Set up containerised development environment to easily reproduce results
- [x] Set up Discord server for project
- [ ] Add other paper authors as repo contributors (in progress)
    - [x] Add Prof. van Zyl
    - [ ] Add Andrew
- [x] Run simple test to see if containerised environment setup works properly
    - [x] Training of RL models with CNN policy network
    - [x] Backtesting trained RL models with CNN policy network
- [ ] Rename REINFORCE to FRONTIER (folder names, script imports, documentation, etc.)
- [ ] Reorganise folders and files to improve repo readability

### Add changes to extend study
- [ ] Branch off main to add new features for NCAA paper
- [ ] Disable transaction costs and investor preferences and run a simple long-only test to maximise portfolio returns to see how it compares to MA-FDRNN, DDPG, PPO, etc.
- [ ] Update transaction cost function to accomodate short trades - see Boyd et al. (2017)
- [ ] Update transaction cost faunction in reward signal
- [ ] Update activation function of policy network output layer to allow short positions (and make sure weights sum to 1)
- [ ] Run a couple of tests to confirm transaction costs and portfolio balances are modelled correctly
- [ ] Rerun study experiments on different markets to see how performance is impacted after allowing short trades

### Publish paper and code
- [ ] Write paper
- [ ] Include detailed README or documentation for steps to use repo and reproduce results
- [ ] Publish paper and open source this repo
