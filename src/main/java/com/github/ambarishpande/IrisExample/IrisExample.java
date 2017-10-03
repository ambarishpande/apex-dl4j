package com.github.ambarishpande.IrisExample;

import org.apache.hadoop.conf.Configuration;
import com.github.ambarishpande.MasterWorkerModule.Dl4jTrainingModule;

import com.datatorrent.api.DAG;
import com.datatorrent.api.StreamingApplication;
import com.datatorrent.api.annotation.ApplicationAnnotation;

/**
 * Iris Dataset Training Example
 *
 * Created by @ambarishpande on 8/4/17.
 */

@ApplicationAnnotation(name = "IrisExample")
public class IrisExample implements StreamingApplication
{
  @Override
  public void populateDAG(DAG dag, Configuration configuration)
  {

    DataSenderOperator inputData = dag.addOperator("inputData", DataSenderOperator.class);
    Dl4jTrainingModule trainingModule = dag.addModule("TrainingModule", Dl4jTrainingModule.class);
    dag.addStream("Data:InputData-Master", inputData.outputData, trainingModule.input);


    /**
     * DL4j Configurations
     *
     * Example of setting configuration from code.
     *
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
      .seed(123)
      .iterations(2)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.0015)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(4).nOut(10)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .build())
      .layer(1, new DenseLayer.Builder().nIn(10).nOut(20)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .build())
      .layer(2, new DenseLayer.Builder().nIn(20).nOut(10)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.SIGMOID)
        .build())
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.SOFTMAX).weightInit(WeightInit.UNIFORM)
        .nIn(10).nOut(3).build())
      .pretrain(false).backprop(true).build();

    trainingModule.setConf(conf);
     */

  }
}
