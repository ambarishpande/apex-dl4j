package com.github.ambarishpande.MasterWorkerModule;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.apache.hadoop.conf.Configuration;

import com.github.ambarishpande.IrisExample.DataSenderOperator;

import com.datatorrent.api.Context;
import com.datatorrent.api.DAG;
import com.datatorrent.api.StreamingApplication;
import com.datatorrent.api.annotation.ApplicationAnnotation;
import com.datatorrent.common.partitioner.StatelessPartitioner;
import com.datatorrent.common.util.DefaultDelayOperator;

/**
 * Created by hadoopuser on 14/1/17.
 */
@ApplicationAnnotation(name = "MasterWorkerModule")
public class MasterWorkerModule implements StreamingApplication
{
  @Override
  public void populateDAG(DAG dag, Configuration configuration)
  {

    int numWorkers = Integer.parseInt(configuration.get("dt.application.MasterWorkerModule.operator.ParameterAverager.prop.numWorkers"));
//      Add all operators
    DataSenderOperator inputData = dag.addOperator("inputData", DataSenderOperator.class);
    Dl4jMasterOperator Master = dag.addOperator("Master", Dl4jMasterOperator.class);
    Dl4jWorkerOperator Worker = dag.addOperator("Worker", Dl4jWorkerOperator.class);
    Dl4jParameterAverager ParameterAverager = dag.addOperator("ParameterAverager", Dl4jParameterAverager.class);
    DefaultDelayOperator delay = dag.addOperator("Delay", DefaultDelayOperator.class);

//    Set Operator Attributes

    RoundRobinStreamCodec rrCodec = new RoundRobinStreamCodec();
    rrCodec.setN(numWorkers);

    dag.setOperatorAttribute(Worker, Context.OperatorContext.PARTITIONER, new StatelessPartitioner<Dl4jWorkerOperator>(numWorkers));
    dag.setInputPortAttribute(Worker.dataPort, Context.PortContext.STREAM_CODEC, rrCodec);

//    Add all Streams

    dag.addStream("Data:Input-Master", inputData.outputData, Master.dataPort).setLocality(DAG.Locality.CONTAINER_LOCAL);
    dag.addStream("Data:Master-Worker", Master.outputData, Worker.dataPort);
    dag.addStream("Parameters:Master-Worker", Master.newParameters, Worker.controlPort);
    dag.addStream("Parameters:Worker-ParameterAverager", Worker.output, ParameterAverager.inputPara);
    dag.addStream("Parameters:ParameterAverager-Delay", ParameterAverager.outputPara, delay.input);
    dag.addStream("Parameters:Delay-Master", delay.output, Master.finalParameters);

    //    DL4j Configurations

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

    Master.setConf(conf);
    Worker.setConf(conf);

//    ParameterAverager.setNumWorkers(numWorkers);
//    Worker.setBatchSize(16);
////
//    Master.setFilename("iris");
//    saver.setConf(conf);
//    Master.setSaveLocation("/home/hadoopuser/iris/");
//    Master.setFilename("iris1.zip" );
//    saver.setFilename("letter-1.zip");

  }
}
