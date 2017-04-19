package com.github.ambarishpande.MasterWorkerModule;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ScheduledExecutorService;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
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
  private transient volatile boolean execute;
  private static int DEFAULT_QUEUE_CAPACITY = 4 * 1024;
  private int queueCapacity = DEFAULT_QUEUE_CAPACITY;
  private transient ScheduledExecutorService scanService;
  private String filename;

  public MultiLayerConfiguration getConf()
  {
    return conf;
  }

  public void setConf(MultiLayerConfiguration conf)
  {
    this.conf = conf;
  }

  private MultiLayerConfiguration conf;
  protected transient LinkedBlockingDeque<INDArrayWrapper> emitQueue;
  private static final Logger LOG = LoggerFactory.getLogger(Dl4jEvaluatorOperator.class);
  @FieldSerializer.Bind(JavaSerializer.class)
  private DataSetIterator dataSetIterator;
  public transient DefaultInputPort<INDArrayWrapper> modelInput = new DefaultInputPort<INDArrayWrapper>()
  {
    @Override
    public void process(INDArrayWrapper model)
    {

      LOG.info("Trained model received by evaluator.." + model.toString());
      emitQueue.add(model);

    }
  };

  public void setup(Context.OperatorContext context)
  {
    lastAccuracy = 0.0;
    dataSetIterator = new IrisDataSetIterator(1, 150);
    emitQueue = new LinkedBlockingDeque<>(queueCapacity);
    execute = true;
    Eval e = new Eval();

    Thread t = new Thread(e);
    t.start();

  }

  public void setFilename(String filename)
  {
    this.filename = filename;
  }

  public class Eval implements Runnable
  {
    INDArrayWrapper params;
    MultiLayerNetwork model = new MultiLayerNetwork(conf);

    @Override
    public void run()
    {
      while (execute) {
        if (!emitQueue.isEmpty()) {
          LOG.info("Queue size : " + emitQueue.size());
          params = emitQueue.removeFirst();
          model.setParams(params.getIndArray());
          LOG.info("Evaluating model " + params.getIndArray().toString());
          dataSetIterator.reset();
          Evaluation eval = new Evaluation(3); //create an evaluation object with 3 possible classes
          while (dataSetIterator.hasNext()) {
            DataSet next = dataSetIterator.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
          }
          LOG.info(eval.stats());
          Configuration configuration = new Configuration();

          try {
            LOG.info("Trying to save model...");
            FileSystem hdfs = FileSystem.newInstance(new URI("hdfs://master:54310/"), configuration);
            FSDataOutputStream hdfsStream = hdfs.create(new Path("/user/hadoopuser/" + filename));
            ModelSerializer.writeModel(model, hdfsStream, true);
            LOG.info("Model saved to location");

          } catch (IOException e) {
//        e.printStackTrace();

            File locationToSave = new File(filename);
//        File locationToSaveBackup = new File("mnist_backup.zip");
            boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
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

      }
    }
  }
}
