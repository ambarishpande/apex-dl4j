package com.github.ambarishpande.MasterWorkerModule;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

/**
 * Created by hadoopuser on 30/1/17
 * Wrapper Class for Network Model.
 */
public class  ApexMultiLayerNetwork
{
  private static final Logger LOG = LoggerFactory.getLogger(ApexMultiLayerNetwork.class);

  private MultiLayerConfiguration conf;
  @FieldSerializer.Bind(JavaSerializer.class)
  private MultiLayerNetwork model;
  private double score;
  private int count;

  public ApexMultiLayerNetwork(){
    LOG.info("Default Constructor called...");
  }


  public ApexMultiLayerNetwork(MultiLayerConfiguration conf)
  {
    LOG.info("Parameterized Constructor called...");
    count = 0;
    this.conf = conf;
    this.model = new MultiLayerNetwork(conf);
    model.init();

   }

  public ApexMultiLayerNetwork(MultiLayerNetwork network)
  {
    this.model = network;
    this.conf = network.getLayerWiseConfigurations();
  }

  public void fit(DataSetWrapper dataSet)
  {
    model.fit(dataSet.getDataSet());
    count++;
  }

  public MultiLayerConfiguration getConf()
  {
    return conf;
  }

  public void setConf(MultiLayerConfiguration conf)
  {
    this.conf = conf;
  }

  public MultiLayerNetwork getModel()
  {
    return model;
  }

  public void setModel(MultiLayerNetwork model)
  {
    this.model = model;
  }

  public double getScore()
  {
    return score;
  }

  public void setScore(double score)
  {
    this.score = score;
  }

  public int getCount()
  {
    return count;
  }

  public void setCount(int count)
  {
    this.count = count;
  }

}
