package com.github.ambarishpande.ScoringExample;

import org.apache.hadoop.conf.Configuration;

import com.github.ambarishpande.IrisExample.DataSenderOperator;
import com.github.ambarishpande.MasterWorkerModule.Dl4jScoringOperator;

import com.datatorrent.api.DAG;
import com.datatorrent.api.StreamingApplication;
import com.datatorrent.api.annotation.ApplicationAnnotation;
import com.datatorrent.lib.io.ConsoleOutputOperator;

/**
 * Created by ambarish on 25/9/17.
 */
@ApplicationAnnotation(name = "ScoringExample")
public class ScoringExample implements StreamingApplication
{
  @Override
  public void populateDAG(DAG dag, Configuration configuration)
  {
    DataSenderOperator dataSenderOperator = dag.addOperator("Data", new DataSenderOperator());
    Dl4jScoringOperator scorer = dag.addOperator("Scorer",new Dl4jScoringOperator());
    ConsoleOutputOperator console = dag.addOperator("Console", new ConsoleOutputOperator());

    dag.addStream("Score",dataSenderOperator.outputData, scorer.input);
    dag.addStream("Classified", scorer.scoredDataset, console.input);
  }
}
