# Description
FRONTIER is a deep reinforcement learning model for portfolio management that takes investor preferences into account. The original version allows long-only trades (see [this paper](https://doi.org/10.36227/techrxiv.19165745.v1) for more details). This repo will extend the original version to allow short trades as well.

# How to use
1. Clone this repo onto your machine
2. Open development environment using one of the following methods:
    * Open in VS code using the `Remote-Containers` extension
    * Manually reproduce the development environment using the Dockerfile in `.devcontainer/`
    * Install the requiremetns in your local environment by running `pip install -r requirements.txt`
3. Browse through `src/models/frontier.py` to see the details of the environment and model architecture, training, backtesting, etc.
4. Run `notebooks/train_template.ipynb` to see an example of how the models are trained
5. Run `notebooks/backtest_template.ipynb` to see an example of how trained modela are backtested

# Roadmap (TODO)
### Setting up contributors and development environment
- [x] Add code from original study
- [x] Add data folders containing market data
- [x] Set up containerised development environment to easily reproduce results
- [x] Set up Discord server for project
- [ ] Add other paper authors as repo contributors (in progress)
    - [x] Add Prof. van Zyl
    - [ ] Add Andrew
- [x] Run simple test on small portfolio (5 assets) to see if containerised environment setup works properly
    - [x] Training of RL models with CNN policy network
    - [x] Backtesting trained RL models with CNN policy network
- [x] Reorganise folders and files to improve repo readability
    - [x] Restructure folders, scripts, notebooks, etc.
    - [x] Update contents of notebooks and scripts to respect new folder structure
- [x] Rename REINFORCE to FRONTIER (folder names, script imports, documentation, etc.)
- [ ] Put Quandl key in dotenv, make sure it's in .gitignore, and update loading key in src/utilities/data_manager.py
- [ ] Test all noteboks and scripts to see if everything works as expected after folder restructure and renaming
    - [x] notebooks/train_template.ipynb
    - [x] notebooks/backtest_template.ipynb
    - [x] notebooks/backtest_actions.ipynb
    - [ ] notebooks/get_all_frontiers.ipynb
    - [ ] notebooks/inspect_backtest_actions_template.ipynb
    - [ ] notebooks/data_processor.ipynb
    - [ ] notebooks/get_index_data.ipynb
    - [ ] notebooks/hot-fix.ipynb
    - [ ] src/utilities/train_parallel.py
    - [ ] src/utilities/backtest_parallel.py
    - [ ] src/utilities/inspect_bt_actions_parallel.py
- [x] Add plot preferences config (LaTeX-like font and light/dark theme selector)
    

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
