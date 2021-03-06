package com.expleague.ml.loss;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class LOOL2 extends L2 {
  public LOOL2(final Vec target, final DataSet<?> base) {
    super(target, base);
  }

  @Override
  public double value(final Stat stats) {
    return stats.weight > 1 ? stats.sum/stats.weight : 0;
  }

  @Override
  public double score(final Stat stats) {
    return stats.weight > 1 ? (- stats.sum * stats.sum / stats.weight) * MathTools.sqr(stats.weight / (stats.weight - 1.)) : 0;
  }
}
