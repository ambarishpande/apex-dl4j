package com.github.ambarishpande.MasterWorkerModule;

import com.datatorrent.api.AutoMetric;
import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.api.annotation.OperatorAnnotation;
import com.datatorrent.common.util.BaseOperator;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
  private double startTime;
  private int numOfClasses;
  @AutoMetric
  private int numberOfExamples;
  @AutoMetric
  private double trainingAccuracy;
  @AutoMetric
  private double time;

  public transient DefaultOutputPort<DataSet> outputData = new DefaultOutputPort<DataSet>();
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

