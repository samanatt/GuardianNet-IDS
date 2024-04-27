# GuardianNet

## Overview

GuardianNet is a network intrusion detection system developed to classify network traffic into normal or malicious categories. This README provides instructions for setting up the dataset and running the GuardianNet system.

## Setup

1. **Download Dataset**: Obtain the dataset sources and follow the instructions provided in the `./dataset-original/*/original/README.md` files to download and prepare the datasets. Replace `*` with the specific dataset name (e.g., CICIDS, KDDCUP, UNSW).

2. **Preprocessing**: Run the `./guardian-net/*_preprocessing.py` scripts to generate preprocessed CSV files for training and testing. These files will be created under the `./guardian-net/dataset-processed/` directory.

3. **Run GuardianNet**: Finally, execute the `./guardian-net/guardian_net.py` script with the `--dataset` flag followed by the dataset name. For example:
   ```bash
   python ./guardian-net/guardian_net.py --dataset CICIDS