package com.github.ambarishpande.EvaluatorModule;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import org.apache.hadoop.conf.Configuration;

import com.github.ambarishpande.MasterWorkerModule.ApexMultiLayerNetwork;
import com.github.ambarishpande.MasterWorkerModule.DataSenderOperator;

import com.datatorrent.api.DAG;
import com.datatorrent.api.StreamingApplication;
import com.datatorrent.api.annotation.ApplicationAnnotation;
import com.datatorrent.lib.io.ConsoleOutputOperator;

/**
 * Created by hadoopuser on 27/2/17.
 */
@ApplicationAnnotation(name="Evaluator")
public class Application implements StreamingApplication
{
  @Override
  public void populateDAG(DAG dag, Configuration configuration)
  {
    DataSenderOperator data = dag.addOperator("Data",DataSenderOperator.class);
    GenericEvaluatorOperator eval = dag.addOperator("Eval",GenericEvaluatorOperator.class);

//    eval.readModelFromLocal("iris.zip");
    eval.readModelFromHdfs("/user/hadoopuser/iris.zip");
    eval.setNumClasses(3);

    dag.addStream("Data-Evaluation",data.outputData,eval.input);





  }
}
