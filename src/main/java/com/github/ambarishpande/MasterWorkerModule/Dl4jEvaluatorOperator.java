package com.github.ambarishpande.MasterWorkerModule;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.common.util.BaseOperator;

/**
 * Created by hadoopuser on 25/1/17.
 */
public class Dl4jEvaluatorOperator extends BaseOperator
{

  private double lastAccuracy;
  private static final Logger LOG = LoggerFactory.getLogger(Dl4jEvaluatorOperator.class);
  @FieldSerializer.Bind(JavaSerializer.class)
  private DataSetIterator dataSetIterator;
  public transient DefaultInputPort<ApexMultiLayerNetwork> modelInput = new DefaultInputPort<ApexMultiLayerNetwork>()
  {
    @Override
    public void process(ApexMultiLayerNetwork model)
    {
      dataSetIterator.reset();
      LOG.info("Trained model received by evaluator.."+model.getModel().params().toString());
      Evaluation eval = new Evaluation(3); //create an evaluation object with 10 possible classes
      while (dataSetIterator.hasNext()) {
        DataSet next = dataSetIterator.next();
        INDArray output = model.getModel().output(next.getFeatureMatrix()); //get the networks prediction
        eval.eval(next.getLabels(), output); //check the prediction against the true class

      }
      double accuracy = eval.accuracy();
      if(accuracy > lastAccuracy)
      {
        lastAccuracy = accuracy;
        Configuration configuration = new Configuration();

        try {

          LOG.info("Trying to save model...");
          FileSystem hdfs = FileSystem.newInstance(new URI("hdfs://master:54310/"), configuration);
          FSDataOutputStream hdfsStream = hdfs.create(new Path("/user/hadoopuser/iris.zip"));
          ModelSerializer.writeModel(model.getModel(), hdfsStream, true);
          LOG.info("Model saved to location");

        } catch (IOException e) {
          e.printStackTrace();
        } catch (URISyntaxException e) {
          e.printStackTrace();
        }


      }
      LOG.info(eval.stats());


    }
  };

  public void setup(Context.OperatorContext context)
  {
    lastAccuracy = 0.0;
    dataSetIterator = new IrisDataSetIterator(10, 150);
  }

}
