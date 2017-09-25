package com.github.ambarishpande.IrisExample;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.api.InputOperator;

/**
 * Created by @ambarishpande on 17/1/17.
 */
public class DataSenderOperator implements InputOperator
{
  private static final Logger LOG = LoggerFactory.getLogger(DataSenderOperator.class);
  @FieldSerializer.Bind(JavaSerializer.class)
  private DataSetIterator dataSetIterator;
  private int numEpochs;
  public transient DefaultOutputPort<DataSet> outputData = new DefaultOutputPort<DataSet>();
  @FieldSerializer.Bind(JavaSerializer.class)
  private DataSet d;
  private int count;
  private int tuplesPerWindow;
  private int sendBatchSize;

  @Override
  public void emitTuples()
  {
    if (tuplesPerWindow > count) {
      try {
        d = dataSetIterator.next();
        outputData.emit(d);
        count++;
      } catch (Exception e) {
        numEpochs--;
        if (numEpochs > 0) {
          dataSetIterator.reset();
        }

      }

    }

  }

  @Override
  public void beginWindow(long l)
  {
    count = 0;
  }

  @Override
  public void endWindow()
  {

  }

  @Override
  public void setup(Context.OperatorContext context)
  {
    dataSetIterator = new IrisDataSetIterator(sendBatchSize, 150);
    LOG.info("Number of examples  :" + dataSetIterator.numExamples());
    LOG.info("iris dataset loaded...");
    d = new DataSet();

  }

  @Override
  public void teardown()
  {

  }

  public int getTuplesPerWindow()
  {
    return tuplesPerWindow;
  }

  public void setTuplesPerWindow(int tuplesPerWindow)
  {
    this.tuplesPerWindow = tuplesPerWindow;
  }

  public int getNumEpochs()
  {
    return numEpochs;
  }

  public void setNumEpochs(int numEpochs)
  {
    this.numEpochs = numEpochs;
  }

  public int getSendBatchSize()
  {
    return sendBatchSize;
  }

  public void setSendBatchSize(int sendBatchSize)
  {
    this.sendBatchSize = sendBatchSize;
  }

}
