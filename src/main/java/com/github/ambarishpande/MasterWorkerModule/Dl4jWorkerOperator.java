package com.github.ambarishpande.MasterWorkerModule;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.common.util.BaseOperator;

import org.apache.commons.lang.NullArgumentException;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by @ambarishpande on 14/1/17.
 */
public class Dl4jWorkerOperator extends BaseOperator
{

  private MultiLayerConfiguration conf;
  private MultiLayerNetwork model;

  public transient DefaultInputPort<DataSet> dataPort = new DefaultInputPort<DataSet>()
  {
    @Override
    public void process(DataSet data)
    {
      try {
        if (!model.isInitCalled()) {
          model.init();
        }
        model.fit(data);

      } catch (NullArgumentException e) {
        System.out.println("Null Pointer exception " + e.getMessage());
      }

    }
  };

  public transient DefaultInputPort<INDArray> controlPort = new DefaultInputPort<INDArray>()
  {
    @Override
    public void process(INDArray parameters)
    {
      model.setParams(parameters);
    }
  };

  public transient DefaultOutputPort<INDArray> output = new DefaultOutputPort<INDArray>();

  public void setup(Context.OperatorContext context)
  {
    System.out.println("Setup Running...");
    model = new MultiLayerNetwork(conf);
    model.init();
    System.out.println("Setup Completed...");
    context.
  }

  public void beginWindow()
  {

  }

  public void endWindow()
  {
    INDArray newParams = model.params();
    output.emit(newParams);
    System.out.println("New Parameters given back to Master...");
  }

  public void setConf(MultiLayerConfiguration conf)
  {
    this.conf = conf;
  }

}

