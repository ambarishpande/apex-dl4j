package com.github.ambarishpande.EvaluatorModule;

import java.io.IOException;

import javax.validation.ConstraintViolationException;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import org.apache.hadoop.conf.Configuration;

import com.datatorrent.api.LocalMode;

/**
 * Created by hadoopuser on 17/3/17.
 */
public class ApplicationTest
{

  @Test
  public void testApplication() throws IOException, Exception
  {
    try {
      LocalMode lma = LocalMode.newInstance();

      Configuration conf = new Configuration(false);
      conf.addResource(this.getClass().getResourceAsStream("/META-INF/properties-Evaluator.xml"));
      lma.prepareDAG(new Application(), conf);
      LocalMode.Controller lc = lma.getController();
      lma.cloneDAG();
      lc.run(); // runs for 10 seconds and quits
//      lc.run();
    } catch (ConstraintViolationException e) {
      Assert.fail("constraint violations: " + e.getConstraintViolations());
    }
  }

  @Test
  public void evaluatorTest(){

    DataSetIterator dataSetIterator = new IrisDataSetIterator(10,150);
    Dl4jEvaluatorOperator eval = new Dl4jEvaluatorOperator();
    eval.setModelFilePath("src/test/resources/models/");
    eval.setModelFileName("25-09-17-11-58-06-iris.zip");
    eval.setNumClasses(3);
    eval.setup(null);
    eval.beginWindow(0);
    while (dataSetIterator.hasNext()){
      eval.input.process(dataSetIterator.next());
    }
    eval.endWindow();

  }

}