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
  @FieldSerializer.Bind(JavaSerializer.class)
  private DataSet d;

  @Override
  public void emitTuples()
  {
    try{
      d = dataSetIterator.next();
//      LOG.info("Sending Data : " + d.toString());
      DataSetWrapper dw = new DataSetWrapper(d);
      outputData.emit(dw);

    }catch (Exception e)
    {
      numEpochs--;
      if(numEpochs > 0)
      {
//        Temporary solution for multiple epochs
        LOG.error("Epoch number " + numEpochs + "Complete" + e.getMessage());
        dataSetIterator.reset();
      }

    }


  }

  @Override
  public void beginWindow(long l)
  {

  }

  @Override
  public void endWindow()
  {

  }

  @Override
  public void setup(Context.OperatorContext context)
  {
    dataSetIterator = new IrisDataSetIterator(10, 150);
//    try {
//      dataSetIterator = new RawMnistDataSetIterator(1,100);
//    } catch (IOException e) {
//      e.printStackTrace();
//    }
    numEpochs = 2000;
    LOG.info("Number of examples  :" + dataSetIterator.numExamples());
    LOG.info("iris dataset loaded...");
    d = new DataSet();

  }

  @Override
  public void teardown()
  {

  }
}
