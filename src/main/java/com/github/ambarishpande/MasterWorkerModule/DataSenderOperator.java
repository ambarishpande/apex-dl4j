package com.github.ambarishpande.MasterWorkerModule;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator;
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
  private DataSetIterator dataSetIterator;
  private int numEpochs;
  public transient DefaultOutputPort<DataSetWrapper> outputData = new DefaultOutputPort<DataSetWrapper>();
  public transient DefaultOutputPort<Integer> controlPort  = new DefaultOutputPort<>();
  private int batchSize;
  @FieldSerializer.Bind(JavaSerializer.class)
  private DataSet d;
  private  int count;

  @Override
  public void emitTuples()
  {
    try{

      if(batchSize>0)
      {
        if (dataSetIterator.hasNext())
        {
          d = dataSetIterator.next();
//      LOG.info("Sending Data : " + d.toString());
          DataSetWrapper dw = new DataSetWrapper(d);
          outputData.emit(dw);
          count++;
          batchSize--;
        }else{
          numEpochs--;
          if(numEpochs > 0)
          {
//        Temporary solution for multiple epochs
            LOG.info("Epoch number " + numEpochs + "Complete");
            controlPort.emit(1);
            dataSetIterator.reset();
          }
        }

      }

    }catch (Exception e)
    {
      LOG.error("Exception :" + e.getStackTrace());

    }


  }

  @Override
  public void beginWindow(long l)
  {
    LOG.info("Begin Window in Sender...");
    batchSize = 5;
    count = 0;
  }

  @Override
  public void endWindow()
  {
    LOG.info("Emmited" + count + " tuples in window.");
  }

  @Override
  public void setup(Context.OperatorContext context)
  {
    dataSetIterator = new IrisDataSetIterator(10, 150);
    numEpochs = 2000;
    count = 0;
    LOG.info("Number of examples  :" + dataSetIterator.numExamples());
    LOG.info("iris dataset loaded...");
    d = new DataSet();

  }

  @Override
  public void teardown()
  {

  }
}
