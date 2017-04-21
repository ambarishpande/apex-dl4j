package com.github.ambarishpande.MasterWorkerModule;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.concurrent.LinkedBlockingDeque;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.util.Date;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;


import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.common.util.BaseOperator;

/**
 * Created by hadoopuser on 3/4/17.
 */
public class ModelSaverOperator extends BaseOperator
{
  private static final Logger LOG = LoggerFactory.getLogger(ModelSaverOperator.class);

  private String filename;
  private String saveLocation;

  public transient DefaultInputPort<MultiLayerNetwork> modelInput = new DefaultInputPort<MultiLayerNetwork>()
  {
    @Override
    public void process(MultiLayerNetwork parameters)
    {

      LOG.info("Trained model received by Saver..");

      Configuration  configuration = new Configuration();
      DateFormat df = new SimpleDateFormat("dd-MM-yy-HH-mm-ss");
      Date dateobj = new Date();
      try {
        LOG.info("Trying to save model...");
        FileSystem hdfs = FileSystem.newInstance(new URI(configuration.get("fs.defaultFS")), configuration);
        FSDataOutputStream hdfsStream = hdfs.create(new Path(saveLocation + df.format(dateobj) + "-" + filename));
        ModelSerializer.writeModel(parameters, hdfsStream, false);
        hdfsStream.close();
        LOG.info("Model saved to location");

      } catch (IOException e) {
        File locationToSave = new File(saveLocation + df.format(dateobj) + "-" +filename);
        boolean saveUpdater = false;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        try {

          ModelSerializer.writeModel(parameters, locationToSave, saveUpdater);
        } catch (IOException e1) {
          e1.printStackTrace();
        }

      LOG.info("Model saved locally...");

      } catch (URISyntaxException e) {
        e.printStackTrace();
      }
    }
  };

  public void setup(Context.OperatorContext context)
  {
    saveLocation = "/home/hadoopuser/iris/";
    filename = "irismodel.zip";
  }

  public void setFilename(String filename)
  {
    this.filename = filename;
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

