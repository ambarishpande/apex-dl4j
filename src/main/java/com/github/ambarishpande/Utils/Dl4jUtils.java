package com.github.ambarishpande.Utils;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

/**
 * Created by ambarish on 25/9/17.
 */
public class Dl4jUtils
{
  public static MultiLayerNetwork readModelFromHdfs(String path)
  {

    Path location = new Path(path);
    Configuration configuration = new Configuration();

    try {
      FileSystem hdfs = FileSystem.newInstance(new URI(configuration.get("fs.defaultFS")), configuration);
      if (hdfs.exists(location)) {
        FSDataInputStream hdfsInputStream = hdfs.open(location);
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(hdfsInputStream);
        return restored;
      }
    } catch (IOException e) {
      e.printStackTrace();
    } catch (URISyntaxException e) {
      e.printStackTrace();
    }
    return null;
  }
}
