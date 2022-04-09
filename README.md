# Description
FRONTIER is a deep reinforcement learning model for portfolio management that takes investor preferences into account. The original version allows long-only trades (see [this paper](https://doi.org/10.36227/techrxiv.19165745.v1) for more details). This repo will extend the original version to allow short trades as well.

# Roadmap (TODO)
- [x] Add code from original study
- [x] Add data folders containing market data
- [x] Set up containerised development environment to easily reproduce results
- [ ] Add other paper authors as repo contributors
- [ ] Run simple test to see if containerised environment setup works properly
- [ ] Rename REINFORCE to FRONTIER (folder names, script imports, documentation, etc.)
- [ ] Reorganise folders and files to improve repo readability
- [ ] Branch off main to add new features for NCAA paper
- [ ] Disable transaction costs and investor preferences and run a simple long-only test to maximise portfolio returns to see how it compares to MA-FDRNN, DDPG, PPO, etc.
- [ ] Update transaction cost function to accomodate short trades - see Boyd et al. (2017)
- [ ] Update transaction cost faunction in reward signal
- [ ] Update activation function of policy function output layer to allow short positions (and make sure weights sum to 1)
- [ ] Run a couple of tests to confirm transaction costs and portfolio balances are modelled correctly
- [ ] Rerun tests on different markets to see how performance is impacted after allowing short trades
- [ ] Write paper
- [ ] Include detailed README or documentation for steps to use repo and reproduce results
- [ ] Publish paper and open source this repo
