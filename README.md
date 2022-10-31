# The solution of the test task.

## Prerequisites
* [Poetry](https://python-poetry.org/docs/#installation)

To run the code from the notebook it is needed to install Poetry.
After installation run 
```
poetry install
```
This command will install all the necessary dependencies
including inner package `hw_test` containing several auxiliary functions
used to solve the task.

After installation will be completed you can start
```
jupyter-lab
```
to explore notebook containing solution of the test task which is
located in the `notebook` folder.

## Task description
The dataset in “2022_Test_ML.csv” consists of 4 features (s_mt, s_mq, d, h_p) and of
2 objectives (QW, DP). The goal of the task is to build the model which approximates
and generalizes QW and DP for arbitrary values of s_mt, s_mq, d, h_p within the range
of their variation ( Table below ).

|     | s_mt | s_mq | d   | h_p  |
|-----|------|------|-----|------|
| Min | 0.8  | 0.8  | 1.0 | 4.0  |
| Max | 2.7  | 2.1  | 3.0 | 10.0 |

## Task requirements
1. The task should be solved with the use of Python in the framework of Jupyter
notebook
   * Analyze data to understand the dataset quality and properties of the target
   functions
   * Building the model for target functions. The choice of the regression model
   could be arbitrary but should be justified
   * Assess the approximation and generalization errors of the obtained model.
   Use the coefficient of determination (R 2 ) for the error metrics
   * Show the dependency of the model approximation and generalization
   errors from the total number of points in the dataset

2. The results should be summarized in the PowerPoint presentation containing the
test case description and results
