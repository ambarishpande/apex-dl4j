package com.github.ambarishpande.MasterWorkerModule;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.UUID;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.serializers.JavaSerializer;

public class CustomSerializableStreamCodec<T> extends com.datatorrent.lib.codec.KryoSerializableStreamCodec<T>
{
  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException
  {
    in.defaultReadObject();
    this.kryo = new Kryo();
    this.kryo.setClassLoader(Thread.currentThread().getContextClassLoader());
    this.kryo.register(MultiLayerNetwork.class, new JavaSerializer()); // Register the types along with custom serializers
  }

  private static final long serialVersionUID = 201411031405L;
}