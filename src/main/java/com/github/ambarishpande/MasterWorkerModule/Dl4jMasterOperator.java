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
import java.util.concurrent.LinkedBlockingDeque;

import com.datatorrent.api.AutoMetric;
import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.api.annotation.OperatorAnnotation;
import com.datatorrent.common.util.BaseOperator;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

/**
 * Created by @ambarishpande on 14/1/17.
 */
@OperatorAnnotation(partitionable = false)
public class Dl4jMasterOperator extends BaseOperator
{

  private static final Logger LOG = LoggerFactory.getLogger(Dl4jMasterOperator.class);
  private MultiLayerConfiguration conf;
  @FieldSerializer.Bind(JavaSerializer.class)
  private MultiLayerNetwork model;
  private String filename;
  @AutoMetric
  private String saveLocation;
  private boolean first;
  private Evaluation eval;
  //  private Thread t;
  private double startTime;
  private int numOfClasses;
  @AutoMetric
  private int numberOfExamples;
  @AutoMetric
  private double trainingAccuracy;
  @AutoMetric
  private double time;
  public transient DefaultOutputPort<DataSet> outputData = new DefaultOutputPort<DataSet>();
//  public transient DefaultOutputPort<INDArrayWrapper> modelOutput = new DefaultOutputPort<INDArrayWrapper>();
  public transient DefaultOutputPort<MultiLayerNetwork> modelOutput = new DefaultOutputPort<MultiLayerNetwork>();
  public transient DefaultInputPort<DataSet> dataPort = new DefaultInputPort<DataSet>()
  {
    @Override
    public void process(DataSet dataSet)
    {

      if (first) {
        LOG.info("Training started...");
        startTime = System.currentTimeMillis();
        first = false;
      }
      numberOfExamples++;
      INDArray output = model.output(dataSet.getFeatureMatrix()); //get the networks prediction
      eval.eval(dataSet.getLabels(), output); //check the prediction against the true class
      outputData.emit(dataSet);


    }
  };

  //  Port to send new parameters to worker.
  public transient DefaultOutputPort<MultiLayerNetwork> newParameters = new DefaultOutputPort<MultiLayerNetwork>();

  //New Parameters received from Dl4jParameterAverager - Send new parameters to all the workers.
  public transient DefaultInputPort<MultiLayerNetwork> finalParameters = new DefaultInputPort<MultiLayerNetwork>()
  {
    @Override
    public void process(MultiLayerNetwork newModel)
    {

      model = newModel.clone();
//      model.setParams(averagedParameters.getIndArray());
      trainingAccuracy = eval.accuracy();
      LOG.info("Training Accuracy : " + trainingAccuracy);
      modelOutput.emit(model);
      time = System.currentTimeMillis() - startTime;
      LOG.info("Number of Examples : " + numberOfExamples);
      LOG.info("Time Taken To Train : " + time);
      newParameters.emit(model);
      LOG.info("Averaged Parameters sent to Workers...");

    }
  };

  public void setup(Context.OperatorContext context)
  {

    model = new MultiLayerNetwork(conf);
    model.init();
    first = true;
    eval = new Evaluation(numOfClasses);
    LOG.info("Model initialized in Master...");
    numberOfExamples = 0;

  }

  public void setFilename(String filename)
  {
    this.filename = filename;
  }

  public void saveModel()
  {

    Configuration configuration = new Configuration();
    DateFormat df = new SimpleDateFormat("dd-MM-yy-HH-mm-ss");
    Date dateobj = new Date();
    try {
      LOG.info("Trying to save model...");
      FileSystem hdfs = FileSystem.newInstance(new URI(configuration.get("fs.defaultFS")), configuration);
      FSDataOutputStream hdfsStream = hdfs.create(new Path(saveLocation + df.format(dateobj) + "-" + filename));
      ModelSerializer.writeModel(model, hdfsStream, false);
      hdfsStream.flush();
      hdfsStream.close();
      LOG.info("Model saved to Hdfs");

    } catch (IOException e) {
//        e.printStackTrace();
      File locationToSave = new File(saveLocation + df.format(dateobj) + "-" + filename);
      boolean saveUpdater = false;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
      try {

        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
      } catch (IOException e1) {
        e1.printStackTrace();
      }

      LOG.info("Model saved locally...");

    } catch (URISyntaxException e) {
      e.printStackTrace();
    }

  }

  public void setConf(MultiLayerConfiguration conf)
  {
    this.conf = conf;
  }

  public String getSaveLocation()
  {
    return saveLocation;
  }

  public void setSaveLocation(String saveLocation)
  {
    this.saveLocation = saveLocation;
  }

  public int getNumOfClasses()
  {
    return numOfClasses;
  }

  public void setNumOfClasses(int numOfClasses)
  {
    this.numOfClasses = numOfClasses;
  }

}

