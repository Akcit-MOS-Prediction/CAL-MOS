# CAL-MOS: Cross-layer Aggregation Leveraging Mean Opinion Scores

## Prerequisites
    - Operating System: Ubuntu (tested on Ubuntu 22.04.4 LTS)
    - Conda Environment (tested on Conda version 24.5.0)
    - Python Version: Python 3.10.14
    - Driver Version: 535.161.07 (tested on a A100 80GB) or 535.104.05 (tested on a RTX 5000)
    - CUDA Version: 12.1 or higher

## Instalation

Install the required dependencies using the following command:

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## Model Architecture

<img src="resources/CAL-MOS.png" alt="Model Architecture" style="display: block; margin: 0 auto;" width="600">