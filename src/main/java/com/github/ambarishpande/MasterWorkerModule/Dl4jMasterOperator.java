package com.github.ambarishpande.MasterWorkerModule;

import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.api.annotation.OperatorAnnotation;
import com.datatorrent.common.util.BaseOperator;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by @ambarishpande on 14/1/17.
 */
@OperatorAnnotation(partitionable = false)
public class Dl4jMasterOperator extends BaseOperator
{

  private static final Logger LOG = LoggerFactory.getLogger(Dl4jMasterOperator.class);


  private MultiLayerConfiguration conf;
  private MultiLayerNetwork model;

  public transient DefaultOutputPort<DataSet> outputData = new DefaultOutputPort<DataSet>();
  public transient DefaultOutputPort<MultiLayerNetwork> modelOutput = new DefaultOutputPort<MultiLayerNetwork>();

  public transient DefaultInputPort<DataSet> dataPort = new DefaultInputPort<DataSet>()
  {
    @Override
    public void process(DataSet dataSet)
    {

      //      Send data to workers.
        LOG.info("DataSet received by Master...");
        LOG.info(dataSet.toString());
        outputData.emit(dataSet);



    }
  };

  //  Port to send new parameters to worker.
  public transient DefaultOutputPort<INDArray> newParameters = new DefaultOutputPort<INDArray>();

  //New Parameters received from Dl4jParameterAverager - Send new parameters to all the workers.
  public transient DefaultInputPort<INDArray> finalParameters = new DefaultInputPort<INDArray>()
  {
    @Override
    public void process(INDArray averagedParameters)
    {

      model.setParams(averagedParameters);
      newParameters.emit(averagedParameters);
      LOG.info("Averaged Parameters sent to Workers...");
    }
  };

  public void setup()
  {

    model = new MultiLayerNetwork(conf);

  }

  public void teardown()
  {
    LOG.info("Final Model Parameters : " + model.params().toString());
    modelOutput.emit(model);

  }

  public void setConf(MultiLayerConfiguration conf)
  {
    this.conf = conf;
  }

}
