package com.github.ambarishpande.MasterWorkerModule;

import javax.ws.rs.core.Application;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.apache.hadoop.conf.Configuration;

import com.datatorrent.api.Context;
import com.datatorrent.api.DAG;
import com.datatorrent.api.StreamingApplication;
import com.datatorrent.common.partitioner.StatelessPartitioner;
import com.datatorrent.common.util.DefaultDelayOperator;

/**
 * Created by hadoopuser on 14/1/17.
 */
public class MasterWorkerModule implements StreamingApplication
{
  private MultiLayerConfiguration conf;
  private int numWorkers;

  @Override
  public void populateDAG(DAG dag, Configuration configuration)
  {

    numWorkers = 2;
//      Add all operators
    DataSenderOperator inputData = dag.addOperator("Input Data", DataSenderOperator.class);
    Dl4jMasterOperator Master = dag.addOperator("Master", Dl4jMasterOperator.class);
    Dl4jWorkerOperator Worker = dag.addOperator("Worker", Dl4jWorkerOperator.class);
    Dl4jParameterAverager ParameterAverager = dag.addOperator("Parameter Averager", Dl4jParameterAverager.class);
    DefaultDelayOperator delay = dag.addOperator("Delay", DefaultDelayOperator.class);

//    Set Operator Attributes
    dag.setOperatorAttribute(Worker, Context.OperatorContext.PARTITIONER, new StatelessPartitioner<Dl4jWorkerOperator>(numWorkers));

//    Add all Streams
    dag.addStream("Data:Input-Master", inputData.outputData, Master.dataPort);
    dag.addStream("Data:Master-Worker", Master.outputData, Worker.dataPort);
    dag.addStream("Parameters:Master-Worker", Master.newParameters, Worker.controlPort);
    dag.addStream("Parameters:Worker-ParameterAverager", Worker.output, ParameterAverager.inputPara);
    dag.addStream("Parameters:ParameterAverager-Delay", ParameterAverager.outputPara, delay.input);
    dag.addStream("Parameters:Delay-Master", delay.output, Master.finalParameters);

//    DL4j Configurations

    conf = new NeuralNetConfiguration.Builder()
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

    Master.setConf(conf);
    Worker.setConf(conf);
    ParameterAverager.setNumWorkers(numWorkers);

  }
}
