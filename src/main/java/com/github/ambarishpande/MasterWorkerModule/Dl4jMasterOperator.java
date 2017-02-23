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
import org.nd4j.linalg.api.ops.impl.transforms.Log;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.commons.io.output.ByteArrayOutputStream;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
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
  public transient DefaultOutputPort<DataSetWrapper> outputData = new DefaultOutputPort<DataSetWrapper>();
  public transient DefaultOutputPort<ApexMultiLayerNetwork> modelOutput = new DefaultOutputPort<ApexMultiLayerNetwork>();

  public transient DefaultInputPort<DataSetWrapper> dataPort = new DefaultInputPort<DataSetWrapper>()
  {
    @Override
    public void process(DataSetWrapper dataSet)
    {

      //      Send data to workers.
      LOG.info("DataSet received by Master...");
      LOG.info("Sending Dataset to workers...");
      outputData.emit(dataSet);
      LOG.info("DataSet Sent to Workers...");
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
      LOG.info("Received Model at Master..." + newModel.getModel().params().toString());
//        LOG.info("Fitted over " + newModel.getCount() + " examples.");
      if(newModel.getCount() % 3 == 0)
      {
//        LOG.info("Epoch complete...");
        LOG.info("Sending Model to Evaluator...");
        modelOutput.emit(newModel);
        LOG.info("Model Sent To evaluator.");
      }
//      model.copy(newModel);
      model = newModel;
      LOG.info("Model set in master..." + model.getModel().params().toString());
      LOG.info("Sending model to Workers...");
      newParameters.emit(newModel);
      LOG.info("Model Sent To Workers...");
      LOG.info("Averaged Parameters sent to Workers...");

    }
  };


  public void setup(Context.OperatorContext context)
  {
    model = new ApexMultiLayerNetwork(conf);
    LOG.info("Model initialized in Master...");
// Code for fetching saved model in case master is killed.
//    Path location = new Path("/user/hadoopuser/iris.zip");
//    Configuration configuration = new Configuration();
//
//    try {
//      FileSystem hdfs = FileSystem.newInstance(new URI("hdfs://master:54310/"), configuration);
//      if(hdfs.exists(location))
//      {
//        FSDataInputStream hdfsInputStream = hdfs.open(location);
//        model.setModel(ModelSerializer.restoreMultiLayerNetwork(hdfsInputStream));
//      }
//
//
//    } catch (IOException e) {
//      e.printStackTrace();
//    } catch (URISyntaxException e) {
//      e.printStackTrace();
//    }
//

  }

  public void beginWindow(long windowId)
  {
    if(windowId%10==0)
    {
      LOG.info("Sending Model to Evaluator...");
      modelOutput.emit(model);
      LOG.info("Model Sent To evaluator.");

    }
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
