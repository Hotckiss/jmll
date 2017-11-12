package com.expleague.ml.func.generic;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.FuncC1;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class Sigmoid extends FuncC1.Stub {
  @Override
  public Vec gradient(Vec x) {
    final double exp = Math.exp(-x.get(0));
    return new SingleValueVec(exp > 1/ MathTools.EPSILON ? 0 : exp / (1 + exp) / (1 + exp));
  }

  @Override
  public double value(Vec x) {
    return 1./(1. + Math.exp(-x.get(0)));
  }

  @Override
  public int dim() {
    return 1;
  }
}