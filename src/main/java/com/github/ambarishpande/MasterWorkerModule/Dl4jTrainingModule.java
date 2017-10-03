package com.github.ambarishpande.MasterWorkerModule;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;

import org.apache.hadoop.conf.Configuration;

import com.datatorrent.api.Context;
import com.datatorrent.api.DAG;
import com.datatorrent.api.Module;
import com.datatorrent.common.partitioner.StatelessPartitioner;
import com.datatorrent.common.util.DefaultDelayOperator;

/**
 * Dl4j Training Module
 * Required Properties:
 * 1) numWorkers - Number of Worker operators to be deployed.
 * 2) batchSize - Number of tuples after which Parameter Averaging is to be done.
 * 3) conf - Neural Network Configuration as MultiLayerConfiguration object. (OR)
 * 4) jsonConf - Neural Network Configuration as json.
 * 4) numOfClasses - Number of output classes
 *
 * Created by @ambarishpande on 28/9/17.
 */
public class Dl4jTrainingModule implements Module
{
  public final transient ProxyInputPort<DataSet> input = new ProxyInputPort<>();
  public final transient ProxyOutputPort<MultiLayerNetwork> output = new ProxyOutputPort<>();

  private int numWorkers;
  private int batchSize;
  private MultiLayerConfiguration conf;
  private int numOfClasses;
  private String jsonConf;

  @Override
  public void populateDAG(DAG dag, Configuration configuration)
  {
    if (!jsonConf.equals("")) {
      conf = MultiLayerConfiguration.fromJson(jsonConf);
    }

    Dl4jMasterOperator Master = dag.addOperator("Master", Dl4jMasterOperator.class);
    Dl4jWorkerOperator Worker = dag.addOperator("Worker", Dl4jWorkerOperator.class);
    Dl4jParameterAverager ParameterAverager = dag.addOperator("ParameterAverager", Dl4jParameterAverager.class);
    DefaultDelayOperator delay = dag.addOperator("Delay", DefaultDelayOperator.class);

    input.set(Master.dataPort);
    output.set(Master.modelOutput);

    RoundRobinStreamCodec rrCodec = new RoundRobinStreamCodec();
    rrCodec.setN(numWorkers);

    CustomSerializableStreamCodec<MultiLayerNetwork> codecMLN = new CustomSerializableStreamCodec<MultiLayerNetwork>();
    CustomSerializableStreamCodec<DataSet> codecDataSet = new CustomSerializableStreamCodec<DataSet>();

    dag.setOperatorAttribute(Worker, Context.OperatorContext.PARTITIONER, new StatelessPartitioner<Dl4jWorkerOperator>(numWorkers));
    dag.setInputPortAttribute(Worker.dataPort, Context.PortContext.STREAM_CODEC, rrCodec);

    dag.setInputPortAttribute(ParameterAverager.inputPara, Context.PortContext.STREAM_CODEC, codecMLN);
    dag.setInputPortAttribute(Master.finalParameters, Context.PortContext.STREAM_CODEC, codecMLN);
    dag.setInputPortAttribute(Worker.controlPort, Context.PortContext.STREAM_CODEC, codecMLN);
    dag.setInputPortAttribute(delay.input, Context.PortContext.STREAM_CODEC, codecMLN);

    dag.setInputPortAttribute(Master.dataPort, Context.PortContext.STREAM_CODEC, codecDataSet);

    dag.addStream("Data:Master-Worker", Master.outputData, Worker.dataPort);
    dag.addStream("Parameters:Master-Worker", Master.newParameters, Worker.controlPort);
    dag.addStream("Parameters:Worker-ParameterAverager", Worker.output, ParameterAverager.inputPara);
    dag.addStream("Parameters:ParameterAverager-Delay", ParameterAverager.outputPara, delay.input);
    dag.addStream("Parameters:Delay-Master", delay.output, Master.finalParameters);

    Master.setConf(conf);
    Master.setNumOfClasses(numOfClasses);
    Worker.setConf(conf);
    ParameterAverager.setNumWorkers(numWorkers);

  }

  public int getNumWorkers()
  {
    return numWorkers;
  }

  public void setNumWorkers(int numWorkers)
  {
    this.numWorkers = numWorkers;
  }

  public int getBatchSize()
  {
    return batchSize;
  }

  public void setBatchSize(int batchSize)
  {
    this.batchSize = batchSize;
  }

  public MultiLayerConfiguration getConf()
  {
    return conf;
  }

  public void setConf(MultiLayerConfiguration conf)
  {
    this.conf = conf;
  }

  public int getNumOfClasses()
  {
    return numOfClasses;
  }

  public void setNumOfClasses(int numOfClasses)
  {
    this.numOfClasses = numOfClasses;
  }

  public String getJsonConf()
  {
    return jsonConf;
  }

  public void setJsonConf(String jsonConf)
  {
    this.jsonConf = jsonConf;
  }


}
