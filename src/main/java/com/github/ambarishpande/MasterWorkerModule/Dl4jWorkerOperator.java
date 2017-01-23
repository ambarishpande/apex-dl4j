package com.github.ambarishpande.MasterWorkerModule;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.common.util.BaseOperator;

import org.apache.commons.io.output.ByteArrayOutputStream;
import org.apache.commons.lang.NullArgumentException;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ObjectOutputStream;
import java.util.ArrayList;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

/**
 * Created by @ambarishpande on 14/1/17.
 */
public class Dl4jWorkerOperator extends BaseOperator
{

  private long windowId;
  private static final Logger LOG = LoggerFactory.getLogger(Dl4jWorkerOperator.class);

  private MultiLayerConfiguration conf;
  private MultiLayerNetwork model;
  private boolean hold;
  @FieldSerializer.Bind(JavaSerializer.class)
  private ArrayList<DataSet> buffer;

  private  int workerId;
  public transient DefaultInputPort<DataSetWrapper> dataPort = new DefaultInputPort<DataSetWrapper>()
  {
    @Override
    public void process(DataSetWrapper data)
    {

      LOG.info(data.getDataSet().toString());
      try {

        if (!model.isInitCalled()) {
          model.init();
        }

        if (hold) {
          LOG.info("Storing Data in Buffer...");
          buffer.add(data.getDataSet());
        } else {
          if (!(buffer.isEmpty())) {
            for (DataSet d : buffer) {
              model.fit(d);
              buffer.remove(d);
              LOG.info("Fitting over buffered datasets");
            }
          }
          model.fit(data.getDataSet());
          LOG.info("Fitting over normal dataset...");
        }
      } catch (NullArgumentException e) {
        LOG.error("Null Pointer exception" + e.getMessage());
      }

    }

  };

  public transient DefaultInputPort<INDArrayWrapper> controlPort = new DefaultInputPort<INDArrayWrapper>()
  {
    @Override
    public void process(INDArrayWrapper parameters)
    {

      LOG.info("Parameters received from Master...");
      model.setParams(parameters.getIndArray());
      LOG.info("Resuming Worker " +  workerId);
      hold = false;
    }
  };

  public transient DefaultOutputPort<INDArrayWrapper> output = new DefaultOutputPort<INDArrayWrapper>();

  public void setup(Context.OperatorContext context)
  {
    LOG.info("Setup Started...");
    model = new MultiLayerNetwork(conf);
    model.init();
    hold = false;
    buffer = new ArrayList<DataSet>();
    workerId = context.getId();
    LOG.info(" Worker ID : " + context.getId());
    LOG.info("Setup Completed...");
  }

  public void beginWindow(long windowId)
  {
    //    Do Nothing
    LOG.info("Window Id:" + windowId);
    this.windowId = windowId;
  }

  public void endWindow()
  {

    if (windowId % 10 == 0) {
      INDArray newParams = model.params();
      LOG.info("New Params : " + newParams.toString());
      output.emit(new INDArrayWrapper(newParams));
      hold = true;
      LOG.info("Holding worker " +  workerId);
      LOG.info("New Parameters given to ParameterAverager...");
    }
  }

  public void setConf(MultiLayerConfiguration conf)
  {
    this.conf = conf;
  }

  public MultiLayerNetwork getModel()
  {
    return model;
  }
}

