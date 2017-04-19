package com.github.ambarishpande.MasterWorkerModule;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.common.util.BaseOperator;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * Created by hadoopuser on 25/1/17.
 */
public class Dl4jEvaluatorOperator extends BaseOperator
{

  private double lastAccuracy;
  private transient volatile boolean execute;
  private static int DEFAULT_QUEUE_CAPACITY = 4 * 1024;
  private int queueCapacity = DEFAULT_QUEUE_CAPACITY;
  protected transient LinkedBlockingDeque<ApexMultiLayerNetwork> emitQueue;
  private static final Logger LOG = LoggerFactory.getLogger(Dl4jEvaluatorOperator.class);
  @FieldSerializer.Bind(JavaSerializer.class)
  private DataSetIterator dataSetIterator;
  public transient DefaultInputPort<ApexMultiLayerNetwork> modelInput = new DefaultInputPort<ApexMultiLayerNetwork>()
  {
    @Override
    public void process(ApexMultiLayerNetwork model)
    {

      LOG.info("Trained model received by evaluator.." + model.getModel().params().toString());
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

  public class Eval implements Runnable
  {
    ApexMultiLayerNetwork model;

    @Override
    public void run()
    {
      while (execute) {
        if (!emitQueue.isEmpty()) {
          LOG.info("Queue size : " + emitQueue.size());
          model = emitQueue.removeFirst();
          LOG.info("Evaluating model " + model.getModel().params().toString());
          dataSetIterator.reset();
          Evaluation eval = new Evaluation(3); //create an evaluation object with 3 possible classes
          while (dataSetIterator.hasNext()) {
            DataSet next = dataSetIterator.next();
            INDArray output = model.getModel().output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
          }
          LOG.info(eval.stats());
        }

      }
    }
  }

}
