package com.github.ambarishpande.MasterWorkerModule;

import java.io.File;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.api.annotation.OperatorAnnotation;
import com.datatorrent.common.util.BaseOperator;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.commons.io.output.ByteArrayOutputStream;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

/**
 * Created by @ambarishpande on 14/1/17.
 */
@OperatorAnnotation(partitionable = false)
public class Dl4jMasterOperator extends BaseOperator
{

  private static final Logger LOG = LoggerFactory.getLogger(Dl4jMasterOperator.class);
  private MultiLayerConfiguration conf;
  private ApexMultiLayerNetwork model;
  private ArrayList<DataSetWrapper> evalData;
  public transient DefaultOutputPort<DataSetWrapper> outputData = new DefaultOutputPort<DataSetWrapper>();
  public transient DefaultOutputPort<ApexMultiLayerNetwork> modelOutput = new DefaultOutputPort<ApexMultiLayerNetwork>();

  public transient DefaultInputPort<DataSetWrapper> dataPort = new DefaultInputPort<DataSetWrapper>()
  {
    @Override
    public void process(DataSetWrapper dataSet)
    {

      //      Send data to workers.
      LOG.info("DataSet received by Master...");
      if(evalData.size() < 15)
      {
        evalData.add(dataSet);
      }
      outputData.emit(dataSet);

    }
  };

  //  Port to send new parameters to worker.
  public transient DefaultOutputPort<ApexMultiLayerNetwork> newParameters = new DefaultOutputPort<ApexMultiLayerNetwork>();

  //New Parameters received from Dl4jParameterAverager - Send new parameters to all the workers.
  public transient DefaultInputPort<ApexMultiLayerNetwork> finalParameters = new DefaultInputPort<ApexMultiLayerNetwork>()
  {
    @Override
    public void process(ApexMultiLayerNetwork newModel)
    {
      model.copy(newModel);
      newParameters.emit(newModel);
      LOG.info("Averaged Parameters sent to Workers...");

    }
  };

  public transient DefaultInputPort<Integer> controlPort = new DefaultInputPort<Integer>()
  {
    @Override
    public void process(Integer flag)
    {
      LOG.info("Sending Model for evaluation...");
      if (flag == 1)
      {
        modelOutput.emit(model);
      }
    }
  };
  public void setup(Context.OperatorContext context)
  {
    model = new ApexMultiLayerNetwork(conf);
    evalData = new ArrayList<>(150);
    LOG.info("Model initialized in Master...");
  }

  public void beginWindow(long windowId)
  {

  }

  public void endWindow()
  {

  }

  public void teardown()
  {

  }

  public void setConf(MultiLayerConfiguration conf)
  {
      this.conf = conf;
  }

}
