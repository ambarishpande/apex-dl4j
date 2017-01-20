package com.github.ambarishpande.MasterWorkerModule;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

  public transient DefaultOutputPort<DataSet> outputData = new DefaultOutputPort<DataSet>();
  public DataSet d;
  @Override
  public void emitTuples()
  {
    if(dataSetIterator.next()!=null)
    {
      d = dataSetIterator.next();
      LOG.info("Sending Data : " + d.toString());
      outputData.emit(d);

    }
    else
    {
      LOG.info("End of Dataset...");
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
      dataSetIterator = new IrisDataSetIterator(1,150);
      LOG.info("Number of examples  :" +dataSetIterator.numExamples());
      LOG.info("Iris dataset loaded...");
      d = new DataSet();

  }

  @Override
  public void teardown()
  {

  }
}
