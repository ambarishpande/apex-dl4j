package com.github.ambarishpande.MasterWorkerModule;

import java.util.ArrayList;
import java.util.Iterator;

import org.deeplearning4j.nn.api.Updater;
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
  private ArrayList<ApexMultiLayerNetwork> workers;
  private int initVal;

  public transient DefaultOutputPort<ApexMultiLayerNetwork> outputPara = new DefaultOutputPort<ApexMultiLayerNetwork>();
  public transient DefaultInputPort<ApexMultiLayerNetwork> inputPara = new DefaultInputPort<ApexMultiLayerNetwork>()
  {
    @Override
    public void process(ApexMultiLayerNetwork newModel)
    {
      if (workers.size() != numWorkers) {
        workers.add(newModel);
        LOG.info("Parameters received for Worker : " + workers.size());
      }

      if (workers.size() == numWorkers) {
//        workers.add(newModel);
        LOG.info("Inside elseif");
        INDArray params = Nd4j.zeros(newModel.getModel().params().shape());
        Updater updater = newModel.getModel().getUpdater();
        INDArray state = Nd4j.zeros(updater.getStateViewArray().shape());
        double score = 0.0;
        int count = 0;
        for (ApexMultiLayerNetwork w : workers) {
          params = params.add(w.getModel().params()); // Adding net parameters
          state = state.addi(w.getModel().getUpdater().getStateViewArray().dup());
//          batchsize
          count+= w.getCount();
          score += w.getModel().score();
          LOG.info("Adding Worker Parameters...");
        }
        count = count - (numWorkers-1)*initVal;
        initVal = count;
        workers.clear();

        // Averaging net parameters, updaters and score.
        INDArray averagedPram = params.divi(numWorkers);
        state = state.divi(numWorkers);
        score /= numWorkers;



        // Setting newly calculated parameters, updaters and score.
        newModel.getModel().setParams(averagedPram);
        updater.setStateViewArray(newModel.getModel(),state,false);
        newModel.getModel().setScore(score);
        newModel.setScore(score);
        newModel.setCount(count);

        LOG.info("Parameters averaged : \n" + averagedPram.toString());
        LOG.info("Updaters Averaged : \n"+updater.getStateViewArray().toString());
        LOG.info("Score : \n" + score);

        outputPara.emit(newModel);
        params = null;
        state = null;

        LOG.info("Parameters averaged and sent to Master...");

      }
    }
  };

  public void setup(Context.OperatorContext context)
  {
    LOG.info("Parameter Averager setting up...");
    workers = new ArrayList<ApexMultiLayerNetwork>();
    initVal = 0;
    LOG.info("Worker size at setup : " + workers.size());

  }

  public void setNumWorkers(int n)
  {
    this.numWorkers = n;
  }

}
