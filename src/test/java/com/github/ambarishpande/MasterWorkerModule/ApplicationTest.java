package com.github.ambarishpande.MasterWorkerModule;

import javax.validation.ConstraintViolationException;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.datatorrent.api.LocalMode;
import com.datatorrent.lib.testbench.CountAndLastTupleTestSink;

import org.apache.hadoop.conf.Configuration;

import com.github.ambarishpande.Utils.Dl4jUtils;

/**
 * Created by hadoopuser on 17/1/17.
 */

public class ApplicationTest {

  @Test
  public void testApplication() throws Exception {
    try {
      LocalMode lma = LocalMode.newInstance();

      Configuration conf = new Configuration(false);
      conf.addResource(this.getClass().getResourceAsStream("/META-INF/properties.xml"));
      lma.prepareDAG(new MasterWorkerModule(), conf);
      LocalMode.Controller lc = lma.getController();
      lma.cloneDAG();
      lc.run();
    } catch (ConstraintViolationException e) {
      Assert.fail("constraint violations: " + e.getConstraintViolations());
    }
  }

  @Test
  public void saverOperatorTest(){
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
      .seed(123)
      .iterations(2)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.0015)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(4).nOut(10)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .build())
      .layer(1, new DenseLayer.Builder().nIn(10).nOut(20)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .build())
      .layer(2, new DenseLayer.Builder().nIn(20).nOut(10)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.SIGMOID)
        .build())
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.SOFTMAX).weightInit(WeightInit.UNIFORM)
        .nIn(10).nOut(3).build())
      .pretrain(false).backprop(true).build();
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    Dl4jModelSaverOperator saver = new Dl4jModelSaverOperator();
    saver.setup(null);
    saver.setSaveLocation("/tmp/savemodel/");
    saver.setFilename("model.zip");
    saver.beginWindow(0);
    saver.modelInput.process(model);
    saver.endWindow();

    Assert.assertEquals(model.summary(),Dl4jUtils.readModelFromHdfs("/tmp/savemodel/model.zip").summary());

  }

  @Test
  public void scorerTest(){

    CountAndLastTupleTestSink sink = new CountAndLastTupleTestSink();
    CountAndLastTupleTestSink sink2 = new CountAndLastTupleTestSink();

    DataSetIterator dataSetIterator = new IrisDataSetIterator(1,150);
    Dl4jScoringOperator scorer = new Dl4jScoringOperator();
    scorer.setModelFilePath("src/test/resources/models/");
    scorer.setModelFileName("25-09-17-11-58-06-iris.zip");
    scorer.setup(null);
    scorer.allScores.setSink(sink);
    scorer.bestClass.setSink(sink2);
    scorer.beginWindow(0);
    DataSet x = dataSetIterator.next();
    scorer.input.process(x);
    scorer.feedback.process(dataSetIterator.next());
    scorer.endWindow();
    double[] pros = (double[]) sink.tuple;
    String result = "";
    for (int i = 0; i<pros.length;i++){
      result += pros[i] + " ";
    }

    Assert.assertEquals("0.9873569011688232 0.010931925848126411 0.0017111020861193538 ",result);
    Assert.assertEquals(new Integer(0),(Integer) sink2.tuple);

  }
}