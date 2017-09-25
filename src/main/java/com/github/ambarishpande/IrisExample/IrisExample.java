package com.github.ambarishpande.IrisExample;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.apache.hadoop.conf.Configuration;

import com.github.ambarishpande.MasterWorkerModule.CustomSerializableStreamCodec;
import com.github.ambarishpande.MasterWorkerModule.Dl4jMasterOperator;
import com.github.ambarishpande.MasterWorkerModule.Dl4jParameterAverager;
import com.github.ambarishpande.MasterWorkerModule.Dl4jWorkerOperator;
import com.github.ambarishpande.MasterWorkerModule.Dl4jModelSaverOperator;
import com.github.ambarishpande.MasterWorkerModule.RoundRobinStreamCodec;

import com.datatorrent.api.Context;
import com.datatorrent.api.DAG;
import com.datatorrent.api.StreamingApplication;
import com.datatorrent.api.annotation.ApplicationAnnotation;
import com.datatorrent.common.partitioner.StatelessPartitioner;
import com.datatorrent.common.util.DefaultDelayOperator;

/**
 * Created by hadoopuser on 8/4/17.
 */

@ApplicationAnnotation(name = "IrisExample")
public class IrisExample implements StreamingApplication
{
  @Override
  public void populateDAG(DAG dag, Configuration configuration)
  {

    int numWorkers = Integer.parseInt(configuration.get("dt.application.IrisExample.operator.ParameterAverager.prop.numWorkers"));

    DataSenderOperator inputData = dag.addOperator("inputData", DataSenderOperator.class);
    Dl4jMasterOperator Master = dag.addOperator("Master", Dl4jMasterOperator.class);
    Dl4jWorkerOperator Worker = dag.addOperator("Worker", Dl4jWorkerOperator.class);
    Dl4jParameterAverager ParameterAverager = dag.addOperator("Parameter Averager", Dl4jParameterAverager.class);
    DefaultDelayOperator delay = dag.addOperator("Delay", DefaultDelayOperator.class);
    Dl4jModelSaverOperator saver = dag.addOperator("Saver",Dl4jModelSaverOperator.class);

    RoundRobinStreamCodec rrCodec = new RoundRobinStreamCodec();
    rrCodec.setN(numWorkers);

    CustomSerializableStreamCodec<MultiLayerNetwork> codecMLN = new CustomSerializableStreamCodec<MultiLayerNetwork>();
    CustomSerializableStreamCodec<DataSet> codecDataSet = new CustomSerializableStreamCodec<DataSet>();

    dag.setOperatorAttribute(Worker, Context.OperatorContext.PARTITIONER, new StatelessPartitioner<Dl4jWorkerOperator>(numWorkers));
    dag.setInputPortAttribute(Worker.dataPort, Context.PortContext.STREAM_CODEC, rrCodec);

    dag.setInputPortAttribute(saver.modelInput, Context.PortContext.STREAM_CODEC, codecMLN);
    dag.setInputPortAttribute(ParameterAverager.inputPara, Context.PortContext.STREAM_CODEC, codecMLN);
    dag.setInputPortAttribute(Master.finalParameters, Context.PortContext.STREAM_CODEC, codecMLN);
    dag.setInputPortAttribute(Worker.controlPort, Context.PortContext.STREAM_CODEC, codecMLN);
    dag.setInputPortAttribute(delay.input, Context.PortContext.STREAM_CODEC, codecMLN);

    dag.setInputPortAttribute(Master.dataPort, Context.PortContext.STREAM_CODEC, codecDataSet);
//    dag.setInputPortAttribute(Worker.dataPort, Context.PortContext.STREAM_CODEC, codecDataSet);

    dag.addStream("Data:InputData-Master", inputData.outputData, Master.dataPort);
    dag.addStream("Data:Master-Worker", Master.outputData, Worker.dataPort);
    dag.addStream("Model:Master-Saver",Master.modelOutput,saver.modelInput);
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

    ParameterAverager.setNumWorkers(numWorkers);

  }
}
