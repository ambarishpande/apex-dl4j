package com.github.ambarishpande.EvaluatorModule;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;
import com.github.ambarishpande.MasterWorkerModule.DataSetWrapper;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.common.util.BaseOperator;

/**
 * Created by hadoopuser on 27/2/17.
 */
public class GenericEvaluatorOperator extends BaseOperator
{

  private static final Logger LOG = LoggerFactory.getLogger(GenericEvaluatorOperator.class);

  private int numClasses;
  private Evaluation eval;
  @FieldSerializer.Bind(JavaSerializer.class)
  private MultiLayerNetwork model;
  private String hdfsUri;
  public transient DefaultInputPort<DataSetWrapper> input = new DefaultInputPort<DataSetWrapper>()
  {
    @Override
    public void process(DataSetWrapper dataSetWrapper)
    {

      DataSet next = dataSetWrapper.getDataSet();
      INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
      eval.eval(next.getLabels(), output); //check the prediction against the true class
      LOG.info(eval.stats());
    }
  };

  public void setup(Context.OperatorContext context)
  {

    LOG.info("Evaluator Setup...");
    eval = new Evaluation(numClasses); //create an evaluation object with numClasses possible classes
    hdfsUri = "hdfs://master:54310/";
  }

  public int getNumClasses()
  {
    return numClasses;
  }

  public void setNumClasses(int numClasses)
  {
    this.numClasses = numClasses;
  }

  public void setModel(MultiLayerNetwork model)
  {

    LOG.info("Model Set");
    this.model = model;
  }

  public String getHdfsUri()
  {
    return hdfsUri;
  }

  public void setHdfsUri(String hdfsUri)
  {
    this.hdfsUri = hdfsUri;
  }

  public void readModelFromHdfs(String path)
  {
//    Code for fetching saved model in case master is killed.
    Path location = new Path(path);
    Configuration configuration = new Configuration();

    try {
      FileSystem hdfs = FileSystem.newInstance(new URI(hdfsUri), configuration);
      if (hdfs.exists(location)) {
        FSDataInputStream hdfsInputStream = hdfs.open(location);
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(hdfsInputStream);
        this.setModel(restored);

      }

    } catch (IOException e) {
      e.printStackTrace();
    } catch (URISyntaxException e) {
      e.printStackTrace();
    }
  }

  public void readModelFromLocal(String path)
  {

    //Save the model
    File locationToSave = new File(path);      //Where to save the network. Note: the file is in .zip format - can be opened externally
    //Load the model
    MultiLayerNetwork restored;
    try {
      restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
      this.setModel(restored);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
