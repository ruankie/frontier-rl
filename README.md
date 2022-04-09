# Description
FRONTIER is a deep reinforcement learning model for portfolio management that takes investor preferences into account.

# Roadmap (TODO)
- [ ] Add data folders containing market data
- [ ] Run simple test to see if containerised environment setup works properly
- [ ] Reorganise folders and files to improve repo readability
- [ ] Branch off main to add new features for NCAA paper
- [ ] Disable transaction costs and investor preferences and run a simple long-only test to maximise portfolio returns to see how it compares to MA-FDRNN, DDPG, PPO, etc.
- [ ] Update transaction cost function to accomodate short trades - see Boyd et al. (2017)
- [ ] Update transaction cost faunction in reward signal
- [ ] Update activation function of policy function output layer to allow short positions (and make sure weights sum to 1)
- [ ] Run a couple of tests to confirm transaction costs and portfolio balances are modelled correctly
- [ ] Rerun tests on different markets to see how performance is impacted after allowing short trades
- [ ] Write paper
- [ ] Publish paper and open source this repo