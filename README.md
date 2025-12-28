# A Deep Reinforcement Learning Agent for Sketch Drawing

## 🚀 Getting Started

### 1\. Prerequisites

Ensure you have Python 3.9+ installed. Install the required dependencies:

Install the required dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```
### 2\. Training

To train the agent from scratch using the default configuration:

```bash
cd main_train
python run_experiments.py
```

*You can modify hyperparameters directly in `run_experiments.py`.*

### 3\. Evaluation

To visualize the agent drawing and calculate metrics (Precision, Recall, F1-Score):

```bash
python evaluate.py
```

You can also evaluate trained models for test dataset:
```bash
python evaluate_batch.py
```
*You can modify hyperparameters directly in `evaluate.py` or `evaluate_batch.py`.*

You can plot results:
```bash
python plot.py
```
or plot multiple results for comparison:
```bash
python plot_multi.py
```
*You can modify hyperparameters directly in `plot.py` or `plot_multi.py`.*
