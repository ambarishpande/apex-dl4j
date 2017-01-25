package com.github.ambarishpande.MasterWorkerModule;

import java.io.File;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.net.URI;
import java.net.URISyntaxException;

import com.datatorrent.api.Context;
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
  private MultiLayerNetwork model;
  private int numTuples;
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
//     if model is trained - same averagedParameters received more than once.
      newParameters.emit(averagedParameters);
      LOG.info("Averaged Parameters sent to Workers...");
    }
  };

  public void setup(Context.OperatorContext context)
  {

    model = new MultiLayerNetwork(conf);
    model.init();
    LOG.info("Model initialized in Master...");
  }


  public void beginWindow(long windowId)
  {
      if (windowId%15 == 0)
      {
        Configuration configuration = new Configuration();

        try {
          LOG.info("Trying to save model...");
          FileSystem hdfs = FileSystem.newInstance( new URI( "hdfs://master:54310/" ), configuration );
          FSDataOutputStream hdfsStream = hdfs.create(new Path("/user/hadoopuser/iris.zip"));
          ModelSerializer.writeModel(model,hdfsStream, true);
          LOG.info("Model saved to location");
//
        } catch (IOException e) {
          e.printStackTrace();
        } catch (URISyntaxException e) {
          e.printStackTrace();
        }

      }
  }

  public void endWindow()
  {

//    LOG.info("Final Model Parameters : " + model.params().toString());
//    modelOutput.emit(model);
  }

  public void teardown()
  {

  }

  public void setConf(MultiLayerConfiguration conf)
  {
    this.conf = conf;
  }

}
