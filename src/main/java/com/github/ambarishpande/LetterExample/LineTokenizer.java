package com.github.ambarishpande.LetterExample;

/**
 * Created by devraj on 24/2/17.
 */

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.ambarishpande.MasterWorkerModule.DataSetWrapper;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.common.util.BaseOperator;
//import org.nd4j.linalg.dataset.Dataset;

public class LineTokenizer extends BaseOperator
{

  private static final Logger LOG = LoggerFactory.getLogger(LineTokenizer.class);
  private int count;
  private int miniBatchSize;
  public DataSet batch;
  public final transient DefaultOutputPort<DataSetWrapper> output = new DefaultOutputPort<>();
  //public final transient DefaultOutputPort<String> outputNegative = new DefaultOutputPort<>();

  public final transient DefaultInputPort<String> input = new DefaultInputPort<String>()
  {

    @Override
    public void process(String line)
    {
      // TODO Auto-generated method stub

      String csvsplit = ",";
      String[] label1 = line.split(csvsplit);

      int results = label1.length;
      double data[] = new double[16];
      for (int i = 1; i < results; i++) {
        try {
          data[i - 1] = Double.parseDouble(label1[i]);

        } catch (NumberFormatException nfe) {
          //NOTE: write something here if you need to recover from formatting errors
        }
        ;
      }

      double lab[] = new double[26];

      lab[(label1[0].charAt(0) - 'A')] = 1.00;
      INDArray label = Nd4j.create(lab, new int[]{1, 26});
//        System.out.println("Label is " + label);

      INDArray arr = Nd4j.create(data, new int[]{1, 16});
//        System.out.println("Data is :" + arr);
      DataSet d = new DataSet(arr, label);
//        System.out.print(d);
      if (count < miniBatchSize) {
        batch.addRow(d, count);
        count++;
      }
      DataSetWrapper dw = new DataSetWrapper(batch);
      output.emit(dw);
      count = 0;

    }

  };

  @Override
  public void setup(Context.OperatorContext context)
  {
    count = 0;
    batch = new DataSet();
  }

  public void beginWindow(long windowId)
  {

  }

  public void endWindow()
  {

  }

  public int getMiniBatchSize()
  {
    return miniBatchSize;
  }

  public void setMiniBatchSize(int miniBatchSize)
  {
    this.miniBatchSize = miniBatchSize;
  }

}

