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

  private ApexMultiLayerNetwork model;
  private MultiLayerConfiguration conf;
  private boolean hold;
  @FieldSerializer.Bind(JavaSerializer.class)
  private ArrayList<DataSet> buffer;
  private int workerId;
  public transient DefaultInputPort<DataSetWrapper> dataPort = new DefaultInputPort<DataSetWrapper>()
  {
    @Override
    public void process(DataSetWrapper data)
    {

      try {

        if (!model.getModel().isInitCalled()) {
          model.getModel().init();
        }

        if (hold) {
          LOG.info("Storing Data in Buffer...");
          buffer.add(data.getDataSet());
        } else {
          if (buffer.size() != 0) {
            LOG.info("Buffered data size : " + buffer.size());
            for (DataSet d : buffer) {
              model.getModel().fit(d);
              LOG.info("Fitting over buffered datasets");
            }
            buffer.clear();
          }

          model.getModel().fit(data.getDataSet());
          LOG.info("Fitting over normal dataset...");
        }
      } catch (NullArgumentException e) {
        LOG.error("Null Pointer exception" + e.getMessage());
      }

    }

  };

  public transient DefaultInputPort<ApexMultiLayerNetwork> controlPort = new DefaultInputPort<ApexMultiLayerNetwork>()
  {
    @Override
    public void process(ApexMultiLayerNetwork newModel)
    {

      LOG.info("newModel received from Master..." + newModel.getModel().params().toString());
      model.copy(newModel);
      LOG.info("Resuming Worker " + workerId);
      hold = false;

    }
  };

  public transient DefaultOutputPort<ApexMultiLayerNetwork> output = new DefaultOutputPort<ApexMultiLayerNetwork>();

  public void setup(Context.OperatorContext context)
  {
    LOG.info("Setup Started...");
    model = new ApexMultiLayerNetwork(conf);
    hold = false;
    buffer = new ArrayList<DataSet>();
    workerId = context.getId();
    LOG.info(" Worker ID : " + context.getId());
    LOG.info("Setup Completed...");
  }

  public void beginWindow(long windowId)
  {
    LOG.info("Window Id:" + windowId);
    this.windowId = windowId;
  }

  public void endWindow()
  {
//  Need to change the logic for sending for averaging. Use numoftuples insted of window.
    if (windowId % 5 == 0) {
      INDArray newParams = model.getModel().params();
      LOG.info("New Params : " + newParams.toString());
      output.emit(model);
      hold = true;
      LOG.info("Holding worker " + workerId);
      LOG.info("New newModel given to ParameterAverager...");
    }
  }

  public void setConf(MultiLayerConfiguration conf)
  {
    this.conf = conf;
  }

}

