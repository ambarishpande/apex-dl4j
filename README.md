# Deeplearning4j - Apache Apex

This repository contains Apex Operators for training, evaluations and scoring of Deeplearning4j Models.


## Deeplearning4j Scoring Operator


### Operator Objective
Using Deeplearning models trained by Deeplearning4j for machine scoring of incoming tuples.


### <a name="props"></a>Properties of Deeplearning4j Scoring Operator

| **Property** | **Description** | **Type** | **Mandatory** | **Default Value** |
| -------- | ----------- | ---- | ------------------ | ------------- |
| *modelFilePath* | Path where the model zip file is stored on hdfs. e.g. /user/ambarish/iris | String | YES | N/A |
| *modelFileName* | Name of the model file. e.g. iris.zip | String | YES | N/A |

  
### Ports

| **Port** | **Description** | **Type** | **Mandatory** |
| -------- | ----------- | ---- | ------------------ |
| *input* | Tuples to be scored by dl4j model are received at this port. | DataSet | Yes |
| *scoredDataset* | Scored Dataset will be emmited to this port. | DataSet | Yes |
| *feedback* | Model can be extended by giving labelled dataset on this port| DataSet | No |



### Example
Example for Dl4jScoringOperator can be found at: [https://github.com/ambarishpande/dl4j-apex/blob/master/src/main/java/com/github/ambarishpande/ScoringExample/ScoringExample.java](https://github.com/ambarishpande/dl4j-apex/blob/master/src/main/java/com/github/ambarishpande/ScoringExample/ScoringExample.java)


## Deeplearning4j Evaluator Operator

### Operator Objective
This operator is used to evaluate a trained Dl4j model.

### <a name="props"></a>Properties of Deeplearning4j Evaluator Operator

| **Property** | **Description** | **Type** | **Mandatory** | **Default Value** |
| -------- | ----------- | ---- | ------------------ | ------------- |
| *numClasses* |  Number of classes that can be predicted | Integer | YES | N/A |
| *modelFilePath* | Path where the model zip file is stored on hdfs. e.g. /user/ambarish/iris | String | YES | N/A |
| *modelFileName* | Name of the model file. e.g. iris.zip | String | YES | N/A |

### Ports

| **Port** | **Description** | **Type** | **Mandatory** |
| -------- | ----------- | ---- | ------------------ |
| *input* | Test data tuples are received at this port. | DataSet | Yes |

### Example
Example for Dl4jEvaluatorOperator can be found at: [https://github.com/ambarishpande/dl4j-apex/blob/master/src/main/java/com/github/ambarishpande/EvaluatorModule/Application.java](https://github.com/ambarishpande/dl4j-apex/blob/master/src/main/java/com/github/ambarishpande/EvaluatorModule/Application.java)

