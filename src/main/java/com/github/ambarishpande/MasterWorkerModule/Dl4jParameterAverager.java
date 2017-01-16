package com.github.ambarishpande.MasterWorkerModule;

import java.util.ArrayList;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.common.util.BaseOperator;

/**
 * Created by @ambarishpande on 16/1/17.
 */

public class Dl4jParameterAverager extends BaseOperator
{

  private int numWorkers;
  private ArrayList<INDArray> workers;
  private INDArray params;

  public transient DefaultOutputPort<INDArray> outputPara = new DefaultOutputPort<INDArray>();
  public transient DefaultInputPort<INDArray> inputPara = new DefaultInputPort<INDArray>()
  {
    @Override
    public void process(INDArray indArray)
    {
      if (workers.size() != numWorkers) {
        workers.add(indArray);
        System.out.println("Parameters received for Worker : " + workers.size());
      } else if (workers.size() == numWorkers) {
        workers.add(indArray);
        System.out.println("Parameters received for Worker : " + workers.size());
        params = Nd4j.zeros(indArray.shape());
        for (INDArray w : workers) {
          params.add(w);
          workers.remove(w);

        }
        params.divi(numWorkers);
        outputPara.emit(params);
        params = null;
        System.out.println("Parameters averaged and sent to Master...");


      }
    }
  };

  public void setNumWorkers(int n)
  {
    this.numWorkers = n;
  }

}
