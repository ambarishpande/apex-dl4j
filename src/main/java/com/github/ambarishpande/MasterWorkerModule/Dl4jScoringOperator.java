package com.github.ambarishpande.MasterWorkerModule;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;
import com.github.ambarishpande.Utils.Dl4jUtils;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.api.annotation.InputPortFieldAnnotation;
import com.datatorrent.common.util.BaseOperator;

/**
 * Operator for scoring using Dl4j Models
 * Required Properties:
 * 1) modelFilePath - Path where the model zip file is stored on hdfs.
 * 2) modelFileName - Name of the model file.
 *
 * Created by @ambarishpande on 25/9/17.
 */
public class Dl4jScoringOperator extends BaseOperator
{
  private static final Logger LOG = LoggerFactory.getLogger(Dl4jScoringOperator.class);

  @FieldSerializer.Bind(JavaSerializer.class)
  private MultiLayerNetwork model;
  private String modelFilePath;
  private String modelFileName;

  public transient DefaultOutputPort<DataSet> scoredDataset = new DefaultOutputPort<>();

  public transient DefaultInputPort<DataSet> input = new DefaultInputPort<DataSet>()
  {
    @Override
    public void process(DataSet dataSet)
    {
      INDArray scores = model.output(dataSet.getFeatureMatrix());
      double[] probabilities = new double[scores.columns()];
      for (int i = 0; i < scores.columns(); i++) {
        probabilities[i] = scores.getDouble(i);
      }
      DataSet scored = dataSet.copy();
      scored.setLabels(scores);
      scoredDataset.emit(scored);
    }
  };

  @InputPortFieldAnnotation(optional = true)
  public transient DefaultInputPort<DataSet> feedback = new DefaultInputPort<DataSet>()
  {
    @Override
    public void process(DataSet dataSet)
    {
      // TODO This can be a time consuming process so do it async.
      MultiLayerNetwork newModel = model.clone();
      LOG.info("Fitting model with feedback data...");
      newModel.fit(dataSet);
      model = newModel.clone();
      String newModelFile = String.valueOf(System.currentTimeMillis()) + "-" + modelFileName;
      Dl4jUtils.writeModelToHdfs(model, modelFilePath + newModelFile);
      setModelFileName(newModelFile);
      LOG.info("New model stored at " + modelFilePath + modelFileName);
    }
  };

  @Override
  public void setup(Context.OperatorContext context)
  {
    super.setup(context);
    model = Dl4jUtils.readModelFromHdfs(modelFilePath + modelFileName);
  }

  public String getModelFilePath()
  {
    return modelFilePath;
  }

  public void setModelFilePath(String modelFilePath)
  {
    this.modelFilePath = modelFilePath;
  }

  public String getModelFileName()
  {
    return modelFileName;
  }

  public void setModelFileName(String modelFileName)
  {
    this.modelFileName = modelFileName;
  }
}
