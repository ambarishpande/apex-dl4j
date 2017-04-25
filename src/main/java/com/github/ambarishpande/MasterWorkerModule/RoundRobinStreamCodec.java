package com.github.ambarishpande.MasterWorkerModule;

import java.io.IOException;
import java.io.ObjectInputStream;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

import com.datatorrent.lib.codec.KryoSerializableStreamCodec;

/**
 * Created by hadoopuser on 17/2/17.
 */
public class RoundRobinStreamCodec extends KryoSerializableStreamCodec<Object>
{
  private int n = 0;
  int nPartitions;
  @Override
  public int getPartition(Object in) {

    return n++ % nPartitions;    // nPartitions is the number of partitions
  }

  public void setN(int n)
  {
    nPartitions = n;
  }

  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException
  {
    in.defaultReadObject();
    this.kryo = new Kryo();
    this.kryo.setClassLoader(Thread.currentThread().getContextClassLoader());
    this.kryo.register(DataSet.class, new JavaSerializer());
  }
} 