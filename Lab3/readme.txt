# Lab 3 - BP algorithms

## Information

* Course: Understanding Deep Neural Networks
* Teacher: Zhang Yi
* Student:
* ID:

## Files

* `bp_online.m` - the MATLAB code file for online BP algorithm, with XOR-worms example for evaluation
* `bp_batch.m` - the MATLAB code file for batch BP algorithm, with XOR-worms example for evaluation
* `fc.m` - the MATLAB code file for feedforward computation
* `bc.m` - the MATLAB code file for backward computation

## Instructions

### Task 0: implement feedforward and backward computation
1. in `fc.m`, implement the forward computing (in either component or vector form), return both the activation and the net input
2. in `bc.m`, implement the backward computing (in either component or vector form)

### Task 1: implement online BP algorithm
in `bp_online.m`:
1. calculate activations `a1`, `a2`, `a3`, and net input `z2`, `z3`
2. calculate cost function `J`
3. calculate sensitivity `delta3`, `delta2`
4. calculate gradient with respect to weights `dw1`, `dw2`
5. update weights `w1`, `w2`

### Task 2: implement batch BP algorithm
in `bp_batch.m`:
1. calculate activations `a1`, `a2`, `a3`, and net input `z2`, `z3`
2. calculate cost function `J`
3. calculate sensitivity `delta3`, `delta2`
4. cumulate gradient with respect to weights `dw1`, `dw2`
5. update weights `w1`, `w2`

### Assignment 2: Reform two networks to be in standard form 
1. submitted in one page pdf form, for example: assignment_ex2.pdf
2. tips: you can draw the figures in the PPT and save as a pdf file to submit.
## Submission

* Submit all (4) `.m` and an 'assignment2' pdf files in a zip file.
* Submit to email address of TA (ithet1000@163.com)
* Use email title:
    ```
    DNN2019F Lab3 <student-id> <student-name>
    ```
    (Don't keep `<`, `>`, `&gt;`, or `&lt;` in the title. Just fill in your student id and your name.)
    TA will be using a filter to **throw away** any email not following such format.
* Deadline: 2019/10/8 23:59
