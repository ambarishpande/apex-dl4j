package com.github.ambarishpande.MasterWorkerModule;

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
} 