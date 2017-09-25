package com.github.ambarishpande.Utils;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;



/**
 * Created by ambarish on 25/9/17.
 */
public class Dl4jUtils
{
  /**
   * Util method to read saved dl4j model from hdfs.
   * @param path
   * @return
   */
  public static MultiLayerNetwork readModelFromHdfs(String path)
  {

    Path location = new Path(path);
    Configuration configuration = new Configuration();

    try {
      FileSystem hdfs = FileSystem.newInstance(new URI(configuration.get("fs.defaultFS")), configuration);
      if (hdfs.exists(location)) {
        FSDataInputStream hdfsInputStream = hdfs.open(location);
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(hdfsInputStream);
        hdfsInputStream.close();
        hdfs.close();
        return restored;
      }
    } catch (IOException e) {
      e.printStackTrace();
    } catch (URISyntaxException e) {
      e.printStackTrace();
    }
    return null;
  }

  /**
   * Util method to save trained model on hdfs.
   * @param model
   * @param path
   * @return
   */
  public static boolean writeModelToHdfs(MultiLayerNetwork model, String path){
    Configuration  configuration = new Configuration();
    try {
      FileSystem hdfs = FileSystem.newInstance(new URI(configuration.get("fs.defaultFS")), configuration);
      FSDataOutputStream hdfsStream = hdfs.create(new Path(path));
      ModelSerializer.writeModel(model, hdfsStream, false);
      hdfsStream.close();
      hdfs.close();
      return  true;

    } catch (IOException e) {
      e.printStackTrace();

    } catch (URISyntaxException e) {
      e.printStackTrace();
    }

    return false;
  }

  /**
   * Method to return the best probable class.
   * @param probabilities
   * @return
   */
  public static int maxIndex(double[] probabilities) {
    int best = 0;
    for (int i = 1; i < probabilities.length; ++i) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    return best;
  }
}
