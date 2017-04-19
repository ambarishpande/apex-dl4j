package com.github.ambarishpande.EvaluatorModule;

import org.apache.hadoop.conf.Configuration;

import com.github.ambarishpande.IrisExample.DataSenderOperator;
import com.github.ambarishpande.LetterExample.FileInputOp;
import com.github.ambarishpande.LetterExample.LineTokenizer;

import com.datatorrent.api.DAG;
import com.datatorrent.api.StreamingApplication;
import com.datatorrent.api.annotation.ApplicationAnnotation;

/**
 * Created by hadoopuser on 27/2/17.
 */
@ApplicationAnnotation(name = "Evaluator")
public class Application implements StreamingApplication
{
  @Override
  public void populateDAG(DAG dag, Configuration configuration)
  {
    DataSenderOperator data = dag.addOperator("Data", DataSenderOperator.class);
//    FileInputOp inputData = dag.addOperator("fileInput",FileInputOp.class);
//    LineTokenizer tokenizer = dag.addOperator("Tokenizer",LineTokenizer.class);
    GenericEvaluatorOperator eval = dag.addOperator("Eval", GenericEvaluatorOperator.class);

//    eval.readModelFromLocal("iris.zip");
//    eval.readModelFromHdfs("/user/hadoopuser/iris.zip");
    eval.readModelFromLocal("07-04-17-14-59-08-iris-16-2000-2.zip");
    eval.setNumClasses(3);

//    dag.addStream("Data:Input-Tokenizer",inputData.output,tokenizer.input).setLocality(DAG.Locality.CONTAINER_LOCAL);
//    dag.addStream("Data-Evaluation",tokenizer.output,eval.input);
    dag.addStream("Data-Evaluation", data.outputData, eval.input);

  }
}
