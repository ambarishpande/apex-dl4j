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
  private ArrayList<DataSetWrapper> buffer;
  private int workerId;
  private int batchSize;
  private int tuplesPerWindow;
  public transient DefaultInputPort<DataSetWrapper> dataPort = new DefaultInputPort<DataSetWrapper>()
  {
    @Override
    public void process(DataSetWrapper data)
    {
      LOG.info("Dataset received by worker " + workerId);
    tuplesPerWindow++;
      try {

        if (hold) {
          LOG.info("Storing Data in Buffer...");
          buffer.add(data);
        } else {
          if (buffer.size() != 0) {

            LOG.info("Buffered data size : " + buffer.size());
            for (DataSetWrapper d : buffer) {
                model.fit(d);
                LOG.info("Fitting over buffered datasets");
            }

            buffer.clear();
          }

          model.fit(data);
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
//      model.copy(newModel);
      model = newModel;
      LOG.info("Model set in worker ..."+ model.getModel().params().toString());
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
    buffer = new ArrayList<DataSetWrapper>();
    workerId = context.getId();
    batchSize = 5;
    LOG.info(" Worker ID : " + context.getId());
    LOG.info("Setup Completed...");
  }

  public void beginWindow(long windowId)
  {
    LOG.info("Window Id:" + windowId);
    this.windowId = windowId;
    tuplesPerWindow = 0;

  }

  public void endWindow()
  {
//  Need to change the logic for sending for averaging. Use numoftuples insted of window.
    LOG.info("Tuples in window " + windowId + " :" + tuplesPerWindow);
    if(windowId%3==0)
      {
//        INDArray newParams = model.getModel().params();
//        LOG.info("New Params : " + newParams.toString());
        LOG.info("Sending Model to Parameter Averager...");
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

