package com.expleague.ml.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.BinOptimizedModel;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.BFGrid;
import com.expleague.ml.impl.BinaryFeatureImpl;

/**
 * User: noxoomo
 */
public  class BumpyRegion extends BinOptimizedModel.Stub {
  public final Vec inside;
  public final BFGrid.Feature[] features;
  public final BFGrid grid;

  public BumpyRegion(final BFGrid grid, BFGrid.Feature[] features, final Vec inside) {
    this.grid = grid;
    this.features = features;
    this.inside = inside;
  }

  @Override
  public final double value(final BinarizedDataSet bds, final int pindex) {
    return cumSum(nonZero(bds, pindex));
  }

  @Override
  public final double value(final Vec x) {
    return cumSum(nonZero(x));
  }

  public final double cumSum(double nz) {
    assert (nz < inside.dim());
    double result = inside.get(0);
    for (int i = 1; i <= nz; ++i) {
      result += inside.get(i);
    }
    return result;
  }

  @Override
  public final int dim() {
    return grid.rows();
  }

  public int nonZero(Vec x) {
    for (int i = 0; i < features.length; i++) {
      if (!features[i].value(x)) {
        return i;
      }
    }
    return features.length;
  }

  public int nonZero(BinarizedDataSet bds, int pindex) {
    for (int i = 0; i < features.length; i++) {
      if (!(bds.bins(features[i].findex())[pindex] > features[i].bin())) {
        return i;
      }
    }
    return features.length;
  }

}

