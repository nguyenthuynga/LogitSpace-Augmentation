# LogitSpace-Augmentation

This repository explores various augmentation methods applied to logit spaces for improving model performance and robustness. Logit space transformations can enhance the generalization capability of models, particularly in challenging classification tasks.

Try logit space augmentation method on Cifar-10 and Cifar 100 following this paper DOI:10.1007/978-3-030-87240-3_45

The augmentation was applied on logit space (the space just before the activation function) instead of output space.

## Overview

Augmentation in logit spaces involves modifying the outputs of the neural network (logits) to introduce diversity and robustness before the final classification step. This project investigates different strategies for logit augmentation and evaluates their impact on model accuracy and robustness.


## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/nguyenthuynga/LogitSpace-Augmentation.git

## Data

Data for Cifar-100 could be found here https://drive.google.com/file/d/1X2REHtXVmqjQTDJe0p3OsaJn6gWTzuMr/view?usp=sharing

