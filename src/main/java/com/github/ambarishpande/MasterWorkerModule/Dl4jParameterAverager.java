package com.github.ambarishpande.MasterWorkerModule;

import java.util.ArrayList;

import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

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
  @FieldSerializer.Bind(JavaSerializer.class)
  private ArrayList<MultiLayerNetwork> workers;
  @FieldSerializer.Bind(JavaSerializer.class)
  private MultiLayerNetwork finalModel;
  public transient DefaultOutputPort<MultiLayerNetwork> outputPara = new DefaultOutputPort<MultiLayerNetwork>();
  public transient DefaultInputPort<MultiLayerNetwork> inputPara = new DefaultInputPort<MultiLayerNetwork>()
  {
    @Override
    public void process(MultiLayerNetwork model)
    {
      if (workers.size() != numWorkers) {
        workers.add(model);
        LOG.info("Parameters received for Worker : " + workers.size());
      }

      if (workers.size() == numWorkers) {

        INDArray params = Nd4j.zeros(model.params().shape());
        INDArray updaterState = Nd4j.zeros(model.getUpdater().getStateViewArray().shape());
        double score = 0.0;
        for (MultiLayerNetwork w : workers) {
          params = params.add(w.params());
          updaterState = updaterState.add(w.getUpdater().getStateViewArray());
          score += w.score();

          LOG.info("Adding Worker Parameters...");
        }
        workers.clear();
        params = params.divi(numWorkers);
        model.setParams(params);

        updaterState = updaterState.divi(numWorkers);
        Updater updater = model.getUpdater();
        updater.setStateViewArray(model, updaterState, false);

        score /= numWorkers;

        model.setScore(score);
        LOG.info("Parameters averaged");

        outputPara.emit(model);

        params = null;
        LOG.info("Parameters averaged and sent to Master...");

      }
    }
  };

  public void setup(Context.OperatorContext context)
  {
    LOG.info("Parameter Averager setting up...");
    workers = new ArrayList<>();
    LOG.info("Worker size at setup : " + workers.size());
  }

  public int getNumWorkers()
  {
    return numWorkers;
  }

  public void setNumWorkers(int n)
  {
    this.numWorkers = n;
  }

}
