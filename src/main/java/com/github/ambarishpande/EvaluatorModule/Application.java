package com.github.ambarishpande.EvaluatorModule;

import org.apache.hadoop.conf.Configuration;

import com.github.ambarishpande.IrisExample.DataSenderOperator;

import com.datatorrent.api.DAG;
import com.datatorrent.api.StreamingApplication;
import com.datatorrent.api.annotation.ApplicationAnnotation;

/**
 * Example Application to use GenericEvaluatorOperator.
 * This app uses a Model Trained on IRIS dataset.
 *
 * Created by hadoopuser on 27/2/17.
 */
@ApplicationAnnotation(name = "Evaluator")
public class Application implements StreamingApplication
{
  @Override
  public void populateDAG(DAG dag, Configuration configuration)
  {
    DataSenderOperator data = dag.addOperator("Data", DataSenderOperator.class);
    GenericEvaluatorOperator eval = dag.addOperator("Eval", GenericEvaluatorOperator.class);
    dag.addStream("Data-Evaluation", data.outputData, eval.input);

  }
}
