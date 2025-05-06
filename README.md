# MANOLO

## Overview
This is the demo code base for MANOLO project. It follows the module framework documented in D1.2 MANOLO_Architecture_and_Benchmarking Framework_v0.3. By checking this demo, you will learn the idea and usage of manalo library.



The module has been tested under python 3.9 and 3.10.

## Release Note
version 0.0.0: Demo purpose.    

## Usage

Please install the package with the command below.
```
pip install -e $YOUR_PATH/UCD_FUN_HORIZON_MANOLO
```
This should allow you to run the function without path referencing.

Here provides a simple test file for you to try. Please run the command below.
```
python manolo_library_demo.py
```
This script will run a quick Cifar10 evaluation on a ResNet model. The dataset will be downloaded to `./manolo/base/data/dataset` when you run it for the first time. Then the script will load the model and execute a quick evaluation (runnable on cpu).   

The progress and the result will be printed as below.
```
 --- CIFAR10 Data Loaded ---
 --- ResNet Loaded --- 
Accuracy on CIFAR-10 testset: 0.00%
```
Please note that this script is only for manolo library demo purpose. It is intended when you see the accuracy is 0% as the pretrained model is not finetuned on Cifar10.


## GitHub Actions

CI testing flow has been added for auto-testing when git commit/push is triggered. The auto-testing flow will confirm if current updates impact the env setup or fail the assertion of the test script `manolo_library_test.py`.

User can review the results of updates in Actions section or the commit history. User should verify their commits are valid and pass the GitHub Actions.
