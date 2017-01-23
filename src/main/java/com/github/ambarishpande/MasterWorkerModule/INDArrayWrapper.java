package com.github.ambarishpande.MasterWorkerModule;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

/**
 * Created by hadoopuser on 23/1/17.
 */
public class INDArrayWrapper
{
  @FieldSerializer.Bind(JavaSerializer.class)
  private INDArray indArray;

  public INDArrayWrapper()
  {

  }

  public INDArrayWrapper(INDArray indArray)
  {
    this.indArray = indArray;
  }

  public INDArray getIndArray()
  {
    return indArray;
  }
}
