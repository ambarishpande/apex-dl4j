package com.github.ambarishpande.MasterWorkerModule;

import java.io.File;
import java.io.IOException;
import java.io.ObjectOutputStream;

import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.api.annotation.OperatorAnnotation;
import com.datatorrent.common.util.BaseOperator;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.commons.io.output.ByteArrayOutputStream;

/**
 * Created by @ambarishpande on 14/1/17.
 */
@OperatorAnnotation(partitionable = false)
public class Dl4jMasterOperator extends BaseOperator
{

  private static final Logger LOG = LoggerFactory.getLogger(Dl4jMasterOperator.class);

  private MultiLayerConfiguration conf;
  private MultiLayerNetwork model;

  public transient DefaultOutputPort<DataSetWrapper> outputData = new DefaultOutputPort<DataSetWrapper>();
  public transient DefaultOutputPort<MultiLayerNetwork> modelOutput = new DefaultOutputPort<MultiLayerNetwork>();

  public transient DefaultInputPort<DataSetWrapper> dataPort = new DefaultInputPort<DataSetWrapper>()
  {
    @Override
    public void process(DataSetWrapper dataSet)
    {

      //      Send data to workers.

      LOG.info("DataSet received by Master..." + dataSet.getDataSet().toString());
      outputData.emit(dataSet);

    }
  };

  //  Port to send new parameters to worker.
  public transient DefaultOutputPort<INDArrayWrapper> newParameters = new DefaultOutputPort<INDArrayWrapper>();

  //New Parameters received from Dl4jParameterAverager - Send new parameters to all the workers.
  public transient DefaultInputPort<INDArrayWrapper> finalParameters = new DefaultInputPort<INDArrayWrapper>()
  {
    @Override
    public void process(INDArrayWrapper averagedParameters)
    {

      model.setParams(averagedParameters.getIndArray());
      newParameters.emit(averagedParameters);
      LOG.info("Averaged Parameters sent to Workers...");
    }
  };

  public void setup()
  {

    model = new MultiLayerNetwork(conf);

  }

  public void beginWindow(long windowId)
  {

  }

  public void endWindow()
  {

//    LOG.info("Final Model Parameters : " + model.params().toString());
//    modelOutput.emit(model);
  }

  public void teardown()
  {
//    File locationToSave = new File("iris.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
//    boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
//    try {
//      ModelSerializer.writeModel(model, locationToSave, saveUpdater);
//    } catch (IOException e) {
//      e.printStackTrace();
//    }
  }

  public void setConf(MultiLayerConfiguration conf)
  {
    this.conf = conf;
  }

}
