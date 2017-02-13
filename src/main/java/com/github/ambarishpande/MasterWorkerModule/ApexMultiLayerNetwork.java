package com.github.ambarishpande.MasterWorkerModule;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

/**
 * Created by hadoopuser on 30/1/17
 * Wrapper Class for Network Model.
 */
public class ApexMultiLayerNetwork
{
  private MultiLayerConfiguration conf;
  @FieldSerializer.Bind(JavaSerializer.class)
  private MultiLayerNetwork model;
  private double score;

  public ApexMultiLayerNetwork(){

  }

  public ApexMultiLayerNetwork(MultiLayerConfiguration conf)
  {
    this.conf = conf;
    this.model = new MultiLayerNetwork(conf);
    model.init();

  }

  public MultiLayerConfiguration getConf()
  {
    return conf;
  }

  public void setConf(MultiLayerConfiguration conf)
  {
    this.conf = conf;
    this.model = new MultiLayerNetwork(conf);
    model.init();
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

  public void copy(ApexMultiLayerNetwork newModel)
  {
    this.model = newModel.getModel();
    this.conf = newModel.getConf();
    this.score = newModel.getScore();
  }
}
