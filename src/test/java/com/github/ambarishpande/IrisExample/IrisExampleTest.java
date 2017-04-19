package com.github.ambarishpande.IrisExample;

import java.io.IOException;

import javax.validation.ConstraintViolationException;

import org.junit.Assert;
import org.junit.Test;

import org.apache.hadoop.conf.Configuration;

import com.datatorrent.api.LocalMode;

/**
 * Created by hadoopuser on 8/4/17.
 */
public class IrisExampleTest
{

  @Test
  public void testApplication() throws IOException, Exception {
    try {
      LocalMode lma = LocalMode.newInstance();

      Configuration conf = new Configuration(false);
      conf.addResource(this.getClass().getResourceAsStream("/META-INF/properties-IrisExample.xml"));
      lma.prepareDAG(new IrisExample(), conf);
      LocalMode.Controller lc = lma.getController();
      lma.cloneDAG();
      lc.run(); // runs for 10 seconds and quits
//      lc.run();
    } catch (ConstraintViolationException e) {
      Assert.fail("constraint violations: " + e.getConstraintViolations());
    }
  }
}