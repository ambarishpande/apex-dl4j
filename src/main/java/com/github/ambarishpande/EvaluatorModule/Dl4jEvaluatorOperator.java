package com.github.ambarishpande.EvaluatorModule;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;
import com.github.ambarishpande.Utils.Dl4jUtils;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.datatorrent.api.AutoMetric;
import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.common.util.BaseOperator;

/**
 * Operator to Evaluate Dl4j Classification models
 * Required Properties
 * 1) numClasses - Number of classes that can be predicted
 * 2) modelFilePath - Path where model file is stored on HDFS
 * 3) modelFileName - Name of the model file (.zip)
 *
 * TODO: Add more metrics for visualizations. Add output ports.
 *
 * Created by @ambarishpande on 27/2/17.
 */
public class Dl4jEvaluatorOperator extends BaseOperator
{
  private static final Logger LOG = LoggerFactory.getLogger(Dl4jEvaluatorOperator.class);

  private int numClasses;
  private String modelFileName;
  private String modelFilePath;

  private Evaluation eval;

  @FieldSerializer.Bind(JavaSerializer.class)
  private transient MultiLayerNetwork model;

  @AutoMetric
  public double accuracy;
  @AutoMetric
  public double precision;
  @AutoMetric
  public double recall;
  @AutoMetric
  public double f1score;

  public transient DefaultInputPort<DataSet> input = new DefaultInputPort<DataSet>()
  {
    @Override
    public void process(DataSet dataSet)
    {

      INDArray output = model.output(dataSet.getFeatureMatrix()); //get the networks prediction
      eval.eval(dataSet.getLabels(), output); //check the prediction against the true class
      LOG.info(eval.stats());
      accuracy = eval.accuracy();
      precision = eval.precision();
      recall = eval.recall();
      f1score = eval.f1();

    }
  };

  public void setup(Context.OperatorContext context)
  {

    LOG.info("Evaluator Setup...");
    eval = new Evaluation(numClasses); //create an evaluation object with numClasses possible classes
    model = Dl4jUtils.readModelFromHdfs(modelFilePath + modelFileName);
  }

  public String getModelFileName()
  {
    return modelFileName;
  }

  public void setModelFileName(String modelFileName)
  {
    this.modelFileName = modelFileName;
  }

  public String getModelFilePath()
  {
    return modelFilePath;
  }

  public void setModelFilePath(String modelFilePath)
  {
    this.modelFilePath = modelFilePath;
  }

  public int getNumClasses()
  {
    return numClasses;
  }

  public void setNumClasses(int numClasses)
  {
    this.numClasses = numClasses;
  }

}
