package com.github.ambarishpande.ScoringExample;

import javax.validation.ConstraintViolationException;

import org.junit.Assert;
import org.junit.Test;

import org.apache.hadoop.conf.Configuration;

import com.github.ambarishpande.MasterWorkerModule.MasterWorkerModule;

import com.datatorrent.api.LocalMode;

import static org.junit.Assert.*;

/**
 * Created by ambarish on 25/9/17.
 */
public class ScoringExampleTest
{
  @Test
  public void testApplication() throws Exception {
    try {
      LocalMode lma = LocalMode.newInstance();

      Configuration conf = new Configuration(false);
      conf.addResource(this.getClass().getResourceAsStream("/META-INF/properties-ScoringExample.xml"));
      lma.prepareDAG(new ScoringExample(), conf);
      LocalMode.Controller lc = lma.getController();
      lma.cloneDAG();
      lc.run();
    } catch (ConstraintViolationException e) {
      Assert.fail("constraint violations: " + e.getConstraintViolations());
    }
  }

}