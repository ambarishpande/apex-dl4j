package com.github.ambarishpande.MasterWorkerModule;

import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.common.util.BaseOperator;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by @ambarishpande on 14/1/17.
 */
public class Dl4jMasterOperator extends BaseOperator
{

  private MultiLayerConfiguration conf;
  private MultiLayerNetwork model;



  private int numWorkers;

  public transient DefaultOutputPort<DataSet> outputData = new DefaultOutputPort<DataSet>();
  public transient DefaultOutputPort<MultiLayerNetwork> modelOutput = new DefaultOutputPort<MultiLayerNetwork>();

  public transient DefaultInputPort<DataSet> dataPort = new DefaultInputPort<DataSet>()
  {
    @Override
    public void process(DataSet dataSet)
    {
      outputData.emit(dataSet);
    }
  };


  public transient  DefaultOutputPort<INDArray> newParameters = new DefaultOutputPort<INDArray>();
  public transient DefaultInputPort<INDArray> finalParameters = new DefaultInputPort<INDArray>()
  {
    @Override
    public void process(INDArray averagedParameters)
    {
      //New Parameters received from Dl4jParameterAverager - Send new parameters to all the workers.

      newParameters.emit(averagedParameters);

    }
  };

  public void setup()
  {
    model = new MultiLayerNetwork(conf);

  }

  public void setNumWorkers(int numWorkers)
  {
    this.numWorkers = numWorkers;
  }

  public void teardown()
  {

  }
}
