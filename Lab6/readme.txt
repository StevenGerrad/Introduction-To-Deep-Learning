# Lab 6 - Big Cat Recognition

## Information

* Course: Understanding Deep Neural Networks
* Teacher: Zhang Yi
* Student:
* ID:

## Files

## Assignment 1

* submit the ppt file named "assignment 1.pptx" and fill in the answer block.
* you can try to answer with digital formulation or take a picture of your handcraft manuscript.

## Assignment 2

* `lab6.m` - the MATLAB code file for the main procedure of this lab.
* `fc.m` - the MATLAB code file for feedforward computation.
* `bc.m` - the MATLAB code file for backward computation
* `lion.zip`, `tiger.zip`, `random_animal.zip` - the zip file of images.
    * Get from https://pan.baidu.com/s/1XSElxpw0tFPLkw8CXKvWnQ Code: `p2x4`

## Instructions

Implement forward computing and backward computing in `fc.m` and `bc.m`.
You can change the interface according to your program need.

Read images from each folder and prepare the dataset.
Each color image have 3 color channels, which leads to a height * width * channels matrix.
Resize each image to a certain size to match the input to your network.
Use all image in `random_animal` as unlabeled set.
Keep 20% of the images in `lion`/`tiger` as testing set, and the rest as training set.
(In fact, you don't need any template any more.)

0. Read and prepare the data
1. Train the autoencoder by using unlabeled data (the unlabeled set)
2. Remove the layers behind sparse representation layer after training
3. Form a new data set in sparse representation layer by using the labeled data set (the trianing set)
4. Form a new training data set for supervised network (the encoded training set and its labels)
5. Training the network by using the new training data set
6. combine the two networks
7. test the network with the testing set

## Submission

* Submit all (3) `.m` files in a zip file. Do not attach `.zip` and `.JPEG` files.
* Submit to email address of TA (ithet1000@163.com)
* Use email title:
    ```
    DNN2019F Lab6 <student-id> <student-name>
    ```
    (Don't keep `<`, `>`, `&gt;`, or `&lt;` in the title. Just fill in your student id and your name.)
    TA will be using a filter to **throw away** any email not following such format.
* Deadline: 2019/11/8 23:59
