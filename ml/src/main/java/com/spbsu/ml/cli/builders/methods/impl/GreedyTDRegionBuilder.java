package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.methods.GreedyTDRegion;
import com.spbsu.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class GreedyTDRegionBuilder implements Factory<VecOptimization> {
  public static Factory<BFGrid> defaultGridBuilder;

  private Factory<BFGrid> gridBuilder = defaultGridBuilder;

  public void setGridBuilder(final Factory<BFGrid> gridBuilder) {
    this.gridBuilder = gridBuilder;
  }

  @Override
  public VecOptimization create() {
    return new GreedyTDRegion(gridBuilder.create());
  }
}
