package com.github.ambarishpande.MasterWorkerModule;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.ambarishpande.Utils.Dl4jUtils;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.common.util.BaseOperator;

/**
 * Operator to save trained Dl4jModel on HDFS
 * Required Properties:
 * 1) saveLocation - Path where model is to be saved.
 * 2) fileName - Name of the model file.
 * Created by @ambarishpande on 3/4/17.
 */
public class Dl4jModelSaverOperator extends BaseOperator
{
  private static final Logger LOG = LoggerFactory.getLogger(Dl4jModelSaverOperator.class);

  private String fileName;
  private String saveLocation;

  public transient DefaultInputPort<MultiLayerNetwork> modelInput = new DefaultInputPort<MultiLayerNetwork>()
  {
    @Override
    public void process(MultiLayerNetwork model)
    {
      LOG.info("Trained model received by Saver..");
      Dl4jUtils.writeModelToHdfs(model, saveLocation + fileName);

    }
  };

  public void setup(Context.OperatorContext context)
  {
  }

  public void setFilename(String fileName)
  {
    this.fileName = fileName;
  }

  public String getSaveLocation()
  {
    return saveLocation;
  }

  public void setSaveLocation(String saveLocation)
  {
    this.saveLocation = saveLocation;
  }

}

