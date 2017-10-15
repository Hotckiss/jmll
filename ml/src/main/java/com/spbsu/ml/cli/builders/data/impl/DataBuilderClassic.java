package com.spbsu.ml.cli.builders.data.impl;

import com.spbsu.commons.util.Pair;
import com.spbsu.ml.cli.builders.data.DataBuilder;
import com.spbsu.ml.cli.builders.data.PoolReader;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;

import java.io.IOException;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class DataBuilderClassic implements DataBuilder {
  private String learnPath;
  private String testPath;
  private PoolReader reader;

  @Override
  public void setLearnPath(final String learnPath) {
    this.learnPath = learnPath;
  }

  @Override
  public void setReader(final PoolReader reader) {
    this.reader = reader;
  }

  public void setTestPath(final String testPath) {
    this.testPath = testPath;
  }


  @Override
  public Pair<Pool, Pool> create() {
    final Pool learn = reader.read(learnPath);
    final Pool test;
    if (testPath != null && !testPath.equals(learnPath)) {
      test = reader.read(testPath);
    }
    else {
      test = learn;
    }
    return Pair.create(learn, test);
  }
}
