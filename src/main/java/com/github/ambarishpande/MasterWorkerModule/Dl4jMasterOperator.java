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
  private MultiLayerNetwork model;
  private ArrayList<DataSetWrapper> evalData;
  private INDArrayWrapper zero;
  private boolean first = true;
  public transient DefaultOutputPort<DataSetWrapper> outputData = new DefaultOutputPort<DataSetWrapper>();
  public transient DefaultOutputPort<MultiLayerNetwork> modelOutput = new DefaultOutputPort<MultiLayerNetwork>();

  public transient DefaultInputPort<DataSetWrapper> dataPort = new DefaultInputPort<DataSetWrapper>()
  {
    @Override
    public void process(DataSetWrapper dataSet)
    {

      if(first)
      {
        DateFormat df = new SimpleDateFormat("dd/MM/yy HH:mm:ss");
        Date dateobj = new Date();
        LOG.info(df.format(dateobj));
        first = false;
      }
      //      Send data to workers.
      LOG.info("DataSet received by Master..." + dataSet.getDataSet().toString());
      if(evalData.size() < 15)
      {
        evalData.add(dataSet);
      }
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

      zero = new INDArrayWrapper(Nd4j.create(averagedParameters.getIndArray().shape()));

      DateFormat df = new SimpleDateFormat("dd/MM/yy HH:mm:ss");
      Date dateobj = new Date();

      if (model.params().eq(averagedParameters.getIndArray()) != zero) {
        LOG.info(df.format(dateobj));
        LOG.info("Training complete...");
        LOG.info("Final Parameters are :" + model.params().toString());
        LOG.info("Evaluation");
        Evaluation eval = new Evaluation(3); //create an evaluation object with 10 possible classes
//        dataSetIterator.reset();
        for (DataSetWrapper d : evalData) {

          INDArrayWrapper output = new INDArrayWrapper(model.output(d.getDataSet().getFeatureMatrix())); //get the networks prediction
          eval.eval(d.getDataSet().getLabels(), output.getIndArray()); //check the prediction against the true class
        }
        LOG.info(eval.stats());


      }
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
    evalData = new ArrayList<>(150);
    LOG.info("Model initialized in Master...");
  }

  public void beginWindow(long windowId)
  {
    if (windowId % 15 == 0) {
      Configuration configuration = new Configuration();

      try {
        LOG.info("Trying to save model...");
        FileSystem hdfs = FileSystem.newInstance(new URI("hdfs://master:54310/"), configuration);
        FSDataOutputStream hdfsStream = hdfs.create(new Path("/user/hadoopuser/iris.zip"));
        ModelSerializer.writeModel(model, hdfsStream, true);
        LOG.info("Model saved to location");

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
