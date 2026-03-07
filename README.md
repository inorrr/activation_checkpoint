# Activation Checkpointing in PyTorch
CS265 Big Data Systems – Systems Project

## Overview

Training modern deep neural networks requires significant GPU memory. A large portion of this memory is consumed by activations (intermediate tensors) generated during the forward pass and later used during backpropagation.

Activation Checkpointing (AC) is a technique that reduces GPU memory usage by trading memory for additional computation. Instead of storing every activation during the forward pass, AC stores only a subset of activations and recomputes the others when they are needed during the backward pass.

This project implements an activation checkpointing system in PyTorch and evaluates the trade-off between memory usage and computation time during neural network training.

The implementation is composed of three major components:

1. Computation Graph Profiler
2. Activation Checkpointing Algorithm
3. Graph Extractor and Rewriter

The system is evaluated using two representative models:

- ResNet-152 (vision model)
- BERT (large language model)

---------------------------------------------------------------------

## Project Goals

The main objectives of this project are:

1. Analyze GPU memory usage during training
2. Identify the role of activations in peak memory consumption
3. Implement an activation checkpointing algorithm
4. Reduce memory consumption by recomputing selected activations
5. Evaluate the trade-off between memory savings and additional computation

The experiments compare training with and without activation checkpointing.

---------------------------------------------------------------------

## Key Idea: Activation Checkpointing

During training:

Forward Pass → store activations  
Backward Pass → use stored activations to compute gradients

Normally, all activations are stored in GPU memory.

Activation checkpointing modifies this process:

Forward Pass → store only selected activations  
Backward Pass → recompute missing activations when needed

This approach reduces peak GPU memory usage but increases computation time due to recomputation. The goal is to find a good balance between memory usage and compute cost.


---------------------------------------------------------------------

## System Components

1. Computation Graph Profiler

The profiler constructs a computation graph for one training iteration including forward pass operations, backward pass operations, and optimizer updates.

The profiler collects the following statistics:

- execution time of each operator
- GPU memory usage
- activation tensor sizes
- first and last use of activations

This information is later used by the checkpointing algorithm to determine which activations should be stored or recomputed.

---------------------------------------------------------------------

2. Activation Checkpointing Algorithm

Using the statistics gathered by the profiler, the activation checkpointing algorithm determines:

- which activations should be stored
- which activations should be discarded and recomputed

The goal of the algorithm is to reduce peak GPU memory usage while minimizing the additional computation cost caused by recomputation.

---------------------------------------------------------------------

3. Graph Extractor and Rewriter

When an activation is discarded during the forward pass, it must be recomputed before the backward pass can compute gradients.

To enable this process:

1. The forward subgraph that produces the activation is extracted
2. The subgraph is replicated
3. The replicated subgraph is inserted into the backward pass before gradient computation

This allows gradients to be computed correctly without storing every activation in memory.

---------------------------------------------------------------------

## Experiments

Experiments are conducted using two models:

- ResNet-152
- BERT

The following metrics are measured during training.

### Memory Usage

Peak GPU memory consumption is measured for different batch sizes. Experiments compare training with and without activation checkpointing.

This produces a graph: Peak Memory vs Batch Size

### Iteration Latency

Training iteration time is measured for different batch sizes.

This produces a graph: Iteration Latency vs Batch Size

These results illustrate the additional computation cost introduced by activation recomputation.


## Installation

Create a Python environment using Conda:

conda create -n dl_project python=3.10
conda activate dl_project

Install project dependencies:

pip install -r requirements.txt

Running Experiments

Example command to run the ResNet experiment:

python experiments/run_resnet_experiment.py

Example command to run the BERT experiment:

python experiments/run_bert_experiment.py

Results and generated figures will be saved in the results directory.
