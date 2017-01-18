package com.github.ambarishpande.MasterWorkerModule;

import java.util.ArrayList;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.api.annotation.OperatorAnnotation;
import com.datatorrent.common.util.BaseOperator;

/**
 * Created by @ambarishpande on 16/1/17.
 */

@OperatorAnnotation(partitionable = false)
public class Dl4jParameterAverager extends BaseOperator
{
  private static final Logger LOG = LoggerFactory.getLogger(Dl4jParameterAverager.class);
  
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
        LOG.info("Parameters received for Worker : " + workers.size());
      }

      if(workers.size() == numWorkers)
      {
//        workers.add(indArray);
        LOG.info("Inside elseif");
        params = Nd4j.zeros(indArray.shape());
        for (INDArray w : workers) {
          params.add(w);
          workers.remove(w);
          LOG.info("Adding Worker Parameters...");
        }
        params.divi(numWorkers);
        LOG.info("Parameters averaged");

        outputPara.emit(params);
        params = null;
        LOG.info("Parameters averaged and sent to Master...");

      }
    }
  };

  public void setup(Context.OperatorContext context)
  {
    LOG.info("Parameter Averager setting up...");
    workers = new ArrayList<INDArray>();
    LOG.info("Worker size at setup : " + workers.size());


  }

  public void setNumWorkers(int n)
  {
    this.numWorkers = n;
  }

}
