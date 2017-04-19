package com.github.ambarishpande.LetterExample;

/**
 * Created by devraj on 24/2/17.
 */

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.hadoop.fs.Path;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.api.annotation.OutputPortFieldAnnotation;
import com.datatorrent.lib.io.fs.AbstractFileInputOperator;

public class FileInputOp extends AbstractFileInputOperator<String>
{
  private static final Logger LOG = LoggerFactory.getLogger(FileInputOp.class);

  /**
   * prefix for file start and finish control tuples
   */
  public static final char START_FILE = '(', FINISH_FILE = ')';

  @OutputPortFieldAnnotation(optional = false)
  public final transient DefaultOutputPort<String> output = new DefaultOutputPort<>();

//    @OutputPortFieldAnnotation(optional = false)
  //  public final transient DefaultOutputPort<String> control = new DefaultOutputPort<>();

  private transient BufferedReader br = null;

  // Path is not serializable so convert to/from string for persistance
  private transient Path filePath;
  private String filePathStr;

  // set to true when end-of-file occurs, to prevent emission of addditional tuples in current window
  private boolean stop;

  // pause for this many milliseconds after end-of-file
  private transient int pauseTime;
  private int tuplesPerWindow;

  @Override
  public void setup(Context.OperatorContext context)
  {
    super.setup(context);

    pauseTime = context.getValue(Context.OperatorContext.SPIN_MILLIS);

    if (null != filePathStr) {      // restarting from checkpoint
      filePath = new Path(filePathStr);
    }
  }

  public void beginWindow(long windowId)
  {
    tuplesPerWindow = 5;
  }

  @Override
  public void endWindow()
  {
    super.endWindow();
    stop = false;
  }

  @Override
  public void emitTuples()
  {

    if (tuplesPerWindow > 0) {        // normal processing
      super.emitTuples();
      tuplesPerWindow--;
      return;
    }

    // we have end-of-file, so emit no further tuples till next window; relax for a bit
    try {
      Thread.sleep(pauseTime);
    } catch (InterruptedException e) {
      LOG.info("Sleep interrupted");
    }

  }

  @Override
  protected InputStream openFile(Path curPath) throws IOException
  {
    LOG.debug("openFile: curPath = {}", curPath);
    filePath = curPath;
    filePathStr = filePath.toString();

    // new file started, send control tuple on control port
    // control.emit(START_FILE + filePath.getName());

    InputStream is = super.openFile(filePath);
    br = new BufferedReader(new InputStreamReader(is));
    return is;
  }

  @Override
  protected void closeFile(InputStream is) throws IOException
  {
    LOG.debug("closeFile: filePath = {}", filePath);
    super.closeFile(is);

    // reached end-of-file, send control tuple on control port
    // control.emit(filePath.getName() + FINISH_FILE);

    br.close();
    br = null;
    filePathStr = null;
    stop = true;
  }

  @Override
  protected String readEntity() throws IOException
  {
    // try to read a line
    final String line = br.readLine();
    if (null != line) {                         // normal case
//            LOG.debug("readEntity: line = {}", line);
      return line;
    }

    // end-of-file (control tuple sent in closeFile()
//        LOG.info("readEntity: EOF for {}", filePath);
    return null;
  }

  @Override
  protected void emit(String tuple)
  {
    output.emit(tuple);
  }

}
