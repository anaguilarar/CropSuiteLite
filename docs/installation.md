# Installation

Follow these steps to set up the **CropSuiteLite** environment on your local machine.

## Prerequisites

Ensure you have the following installed:

*   **Git**: To clone the repository.
*   **Conda** (Anaconda or Miniconda): To manage the Python environment.

## 1. Clone the repository

Clone the repository to your local machine and navigate into the project directory:

```bash
git clone https://github.com/AdaptationAtlas/CropSuiteLite.git
cd CropSuiteLite
```

## 2. Create Python environment using conda

reate a dedicated Conda environment with Python 3.11. This ensures dependencies do not conflict with other projects.

```bash
conda create -n cropsuite python==3.11
conda activate cropsuite
```

## 3. Install dependencies

Install the required Python packages using pip.

```bash
pip install -r requirements.text
```
