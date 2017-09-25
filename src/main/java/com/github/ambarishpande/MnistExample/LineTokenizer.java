package com.github.ambarishpande.MnistExample;

/**
 * Created by devraj on 24/2/17.
 */

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.common.util.BaseOperator;

public class LineTokenizer extends BaseOperator
{

  private int tuplesPerWindow = 5;
  public final transient DefaultOutputPort<DataSet> output = new DefaultOutputPort<>();
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
        double data[] = new double[784];
        for (int i = 1; i < results; i++) {
          try {
            data[i - 1] = Double.parseDouble(label1[i]);
            if(data[i-1] > 35)
            {
              data[i-1] = 1;
            }
            else
            {
              data[i-1] = 0;
            }
          } catch (NumberFormatException nfe) {
            //NOTE: write something here if you need to recover from formatting errors
          }
          ;
        }

        double lab[] = new double[10];

        lab[Integer.parseInt(label1[0])] = 1.00;
        INDArray label = Nd4j.create(lab, new int[]{1, 10});
//            System.out.println("Label is " + label);

        INDArray arr = Nd4j.create(data, new int[]{1, 784});
//            System.out.println("Data is :" + arr);
        DataSet d = new DataSet(arr, label);
///            System.out.print(d);
//        DataSetWrapper dw = new DataSetWrapper(d);
        output.emit(d);



    }

  };

  @Override
  public void setup(Context.OperatorContext context)
  {

  }

  public void beginWindow(long windowId)
  {

  }

}

