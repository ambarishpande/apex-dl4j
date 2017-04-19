package com.github.ambarishpande.MnistExample;

import java.io.IOException;

import javax.validation.ConstraintViolationException;

import org.junit.Assert;
import org.junit.Test;

import org.apache.hadoop.conf.Configuration;

import com.github.ambarishpande.MasterWorkerModule.MasterWorkerModule;

import com.datatorrent.api.LocalMode;

import static org.junit.Assert.*;

/**
 * Created by hadoopuser on 23/3/17.
 */
public class MnistExampleTest
{

  @Test
  public void testApplication() throws IOException, Exception {
    try {
      LocalMode lma = LocalMode.newInstance();

      Configuration conf = new Configuration(false);
      conf.addResource(this.getClass().getResourceAsStream("/META-INF/properties-MnistExample.xml"));
      lma.prepareDAG(new MnistExample(), conf);
      LocalMode.Controller lc = lma.getController();
      lma.cloneDAG();
      lc.run(); // runs for 10 seconds and quits
//      lc.run();
    } catch (ConstraintViolationException e) {
      Assert.fail("constraint violations: " + e.getConstraintViolations());
    }
  }
}