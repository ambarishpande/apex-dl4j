package com.github.ambarishpande.MnistExample;

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
import com.github.ambarishpande.MasterWorkerModule.ModelSaverOperator;
import com.github.ambarishpande.MasterWorkerModule.RoundRobinStreamCodec;

import com.datatorrent.api.Context;
import com.datatorrent.api.DAG;
import com.datatorrent.api.StreamingApplication;
import com.datatorrent.api.annotation.ApplicationAnnotation;
import com.datatorrent.common.partitioner.StatelessPartitioner;
import com.datatorrent.common.util.DefaultDelayOperator;

/**
 * Created by hadoopuser on 23/3/17.
 */
@ApplicationAnnotation(name="MnistExample")
public class MnistExample implements StreamingApplication
{
  private MultiLayerConfiguration conf;
  private int numWorkers;

  @Override
  public void populateDAG(DAG dag, Configuration configuration)
  {

    numWorkers = Integer.parseInt(configuration.get("dt.application.MnistExample.operator.ParameterAverager.prop.numWorkers"));;
//      Add all operators
    FileInputOp inputData = dag.addOperator("fileInput", FileInputOp.class);
    LineTokenizer tokenizer = dag.addOperator("Tokenized Data", LineTokenizer.class);
    Dl4jMasterOperator Master = dag.addOperator("Master", Dl4jMasterOperator.class);
    Dl4jWorkerOperator Worker = dag.addOperator("Worker", Dl4jWorkerOperator.class);
    Dl4jParameterAverager ParameterAverager = dag.addOperator("Parameter Averager", Dl4jParameterAverager.class);
    DefaultDelayOperator delay = dag.addOperator("Delay", DefaultDelayOperator.class);
//    Dl4jEvaluatorOperator eval = dag.addOperator("Eval",Dl4jEvaluatorOperator.class);
    ModelSaverOperator saver = dag.addOperator("Saver",ModelSaverOperator.class);

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

//    Add all Streams
    dag.addStream("Data:Input-Tokenizer", inputData.output, tokenizer.input).setLocality(DAG.Locality.CONTAINER_LOCAL);
    dag.addStream("Data:Tokenizer-Master",tokenizer.output, Master.dataPort);
    dag.addStream("Data:Master-Worker", Master.outputData, Worker.dataPort);
    dag.addStream("Model:Master-Saver",Master.modelOutput,saver.modelInput);
    dag.addStream("Parameters:Master-Worker", Master.newParameters, Worker.controlPort);
    dag.addStream("Parameters:Worker-ParameterAverager", Worker.output, ParameterAverager.inputPara);
    dag.addStream("Parameters:ParameterAverager-Delay", ParameterAverager.outputPara, delay.input);
    dag.addStream("Parameters:Delay-Master", delay.output, Master.finalParameters);
//    DL4j Configurations

    conf = new NeuralNetConfiguration.Builder()
      .seed(123)
      .iterations(2)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.006)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .regularization(true).l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
        .nIn(28 * 28)
        .nOut(1000)
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
        .nIn(1000)
        .nOut(10)
        .activation(Activation.SOFTMAX)
        .weightInit(WeightInit.XAVIER)
        .build())
      .pretrain(false).backprop(true) //use backpropagation to adjust weights
      .build();



    Master.setConf(conf);
    Worker.setConf(conf);

  }
}