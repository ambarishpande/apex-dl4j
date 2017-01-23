package com.github.ambarishpande.MasterWorkerModule;

import org.nd4j.linalg.dataset.DataSet;

import com.esotericsoftware.kryo.serializers.FieldSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

/**
 * Created by hadoopuser on 23/1/17.
 */
public class DataSetWrapper
{

  @FieldSerializer.Bind(JavaSerializer.class)
  private DataSet dataSet;

  public DataSetWrapper()
  {

  }

  public DataSetWrapper(DataSet d)
  {
    this.dataSet = d;
  }

  public DataSet getDataSet()
  {
    return dataSet;
  }

  public void setDataSet(DataSet dataSet)
  {
    this.dataSet = dataSet;
  }

}
