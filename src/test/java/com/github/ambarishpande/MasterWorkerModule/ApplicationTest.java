package com.github.ambarishpande.MasterWorkerModule;

import java.io.IOException;

import javax.validation.ConstraintViolationException;

import org.junit.Assert;
import org.junit.Test;

import com.datatorrent.api.LocalMode;

import org.apache.hadoop.conf.Configuration;

/**
 * Created by hadoopuser on 17/1/17.
 */

public class ApplicationTest {

  @Test
  public void testApplication() throws IOException, Exception {
    try {
      LocalMode lma = LocalMode.newInstance();
      Configuration conf = new Configuration(false);
      conf.addResource(this.getClass().getResourceAsStream("/META-INF/properties.xml"));
      lma.prepareDAG(new MasterWorkerModule(), conf);
      LocalMode.Controller lc = lma.getController();
      lc.run(100000); // runs for 10 seconds and quits
//      lc.run();
    } catch (ConstraintViolationException e) {
      Assert.fail("constraint violations: " + e.getConstraintViolations());
    }
  }

}