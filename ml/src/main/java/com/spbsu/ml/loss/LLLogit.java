package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * We use value representation = \frac{e^x}{e^x + 1}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class LLLogit extends FuncC1.Stub implements TargetFunc {
  private final Vec target;
  private final DataSet<?> owner;

  public LLLogit(Vec target, final DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
  }

  public Vec gradient(Vec x) {
    Vec result = new ArrayVec(x.dim());
    for (int i = 0; i < x.dim(); i++) {
      double expX = exp(x.get(i));
      double pX = expX / (1 + expX);
      if (target.get(i) > 0) // positive example
        result.set(i, pX - 1);
      else // negative
        result.set(i, pX);
    }
    return result;
  }

  @Override
  public int dim() {
    return target.dim();
  }

  public double value(Vec point) {
    double result = 0;
    for (int i = 0; i < point.dim(); i++) {
      double expMX = exp(-point.get(i));
      double pX = 1. / (1. + expMX);
      if (target.get(i) > 0) // positive example
        result += log(pX);
      else // negative
        result += log(1 - pX);
    }

    return exp(result / point.dim());
  }

  public int label(final int idx) {
    return (int)target.get(idx);
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }
}