![GitHub](https://img.shields.io/github/license/ruankie/frontier-rl) 
![GitHub contributors](https://img.shields.io/github/contributors/ruankie/frontier-rl) 
![GitHub last commit](https://img.shields.io/github/last-commit/ruankie/frontier-rl)

# Description
FRONTIER is a deep reinforcement learning model for portfolio management that takes investor preferences into account. The original version allows long-only trades (see [this paper](https://doi.org/10.36227/techrxiv.19165745.v1) for more details). This repo will extend the original version to allow short trades as well.

# Usage 
There are two ways to set up this development environment. One for development with VS Code, and the other for development through a browser interface using Jupyter Lab. Please refer to the relevant section below:

## Running parallelised experiments
### Training
Specify your training configuration settings in `src/config/train_config.json` and run:
```bash
cd src/utilities
python train_parallel.py
```

### Backtesting
Specify your backtesting configuration settings in `src/config/backtest_config.json` and run:
```bash
cd src/utilities
python backtest_parallel.py
```

## Developing in VS Code
*Prerequisites: You must have [Docker](https://docs.docker.com/get-docker/) and [VS Code](https://code.visualstudio.com/download) installed.*
1. Ensure your VS Code has the `Remote-Containers` extension installed
2. Clone this repo
3. Open the root folder using the `Remote-Containers` extension:
   1. Open your command pallette in VS Code (`fn`+`F1`)
   2. Type in and select: `Remote-Containers: Open Folder in Container`
   3. Select the `frontier-rl` folder
   4. Wait for the container to build (first time will take a couple of minutes)
4. Once the container is built, you can start developing.
5. Insert your API keys and desired plot theme into `.env` as per `.env_example`
6. Browse through `src/models/frontier.py` to see the details of the environment and model architecture, training, backtesting, etc.
7. Run `notebooks/train_template.ipynb` to see an example of how the models are trained
8. Run `notebooks/backtest_template.ipynb` to see an example of how trained models are backtested

## Developing in  Jupyter Lab
*Prerequisites: You must have [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed.*
1. Clone this repo
2. Navigate to the `.devcontainer` folder by running `cd frontier-rl/.devcontainer`
3. Build the development environment container by running `docker-compose up` (this will build and configure the environment and start a jupyter lab server inside the notebook which can be accessed remotely)
4. Once done, open a web browser and navigate to the jupyter server using your token. It should be something like `http://127.0.0.1:10000/lab?token=f1731cd54965375ea245efc131ef6c1172f415139e38e8e9`
5. Now you can start developing.
6. Insert your API keys and desired plot theme into `.env` as per `.env_example`
7. Browse through `src/models/frontier.py` to see the details of the environment and model architecture, training, backtesting, etc.
8. Run `notebooks/train_template.ipynb` to see an example of how the models are trained
9. Run `notebooks/backtest_template.ipynb` to see an example of how trained models are backtested


# Roadmap (TODO)
## Setting up contributors and development environment
- [x] Add code from original study
- [x] Add data folders containing market data
- [x] Set up containerised development environment to easily reproduce results
    - [x] Fore development with VS Code through the `Remote-Containers` extension
    - [x] For remote development with Jupyter Lab through jupyter server in container
- [x] Set up Discord server for project
- [x] Add other paper authors as repo contributors (in progress)
    - [x] Add Prof. van Zyl
    - [x] Add Andrew
- [x] Run simple test on small portfolio (5 assets) to see if containerised environment setup works properly
    - [x] Training of RL models with CNN policy network
    - [x] Backtesting trained RL models with CNN policy network
- [x] Reorganise folders and files to improve repo readability
    - [x] Restructure folders, scripts, notebooks, etc.
    - [x] Update contents of notebooks and scripts to respect new folder structure
- [x] Rename REINFORCE to FRONTIER (folder names, script imports, documentation, etc.)
- [x] Put Quandl key in dotenv, make sure it's in .gitignore, and update loading key in src/utilities/data_manager.py
- [x] Test all noteboks and scripts to see if everything works as expected after folder restructure and renaming
    - [x] notebooks/train_template.ipynb
    - [x] notebooks/backtest_template.ipynb
    - [x] notebooks/backtest_actions.ipynb
    - [x] notebooks/get_all_frontiers.ipynb
    - [x] notebooks/inspect_backtest_actions_template.ipynb
    - [x] notebooks/data_preprocessor.ipynb
    - [x] notebooks/get_index_data.ipynb
    - [x] notebooks/hot-fix.ipynb
- [x] Add plot preferences config (LaTeX-like font and light/dark theme selector)
    

## Add changes to extend study
- [x] Branch off main to add new features for NCAA paper
- [ ] Disable transaction costs and investor preferences and run a simple long-only test to maximise portfolio returns to see how it compares to MA-FDRNN, DDPG, PPO, etc.
- [x] Update transaction cost function to accomodate short trades - see Boyd et al. (2017)
- [x] Update transaction cost faunction in reward signal
- [x] Update activation function of policy network output layer to allow short positions (and make sure weights sum to 1)
- [ ] Run a couple of tests to confirm transaction costs and portfolio balances are modelled correctly
- [ ] Rerun study experiments on different markets to see how performance is impacted after allowing short trades

## Publish paper and code
- [ ] Write paper
- [ ] Make docs from docstrings for all code
- [ ] Include detailed README or documentation for steps to use repo and reproduce results
- [ ] Publish paper and open source this repo
