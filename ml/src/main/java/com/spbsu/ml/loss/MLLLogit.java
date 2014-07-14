package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.FuncC1;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * We use value representation = \frac{e^x}{e^x + 1}.
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class MLLLogit extends FuncC1.Stub {
  private final IntSeq target;
  private final int classesCount;

  public MLLLogit(IntSeq target) {
    this.target = target;
    classesCount = target.at(ArrayTools.max(target)) + 1;
  }

  @Override
  public Vec gradient(Vec point) {
    Vec result = new ArrayVec(point.dim());
    Mx resultMx = new VecBasedMx(target.length(), result);
    Mx mxPoint = new VecBasedMx(target.length(), point);
    for (int i = 0; i < target.length(); i++) {
      double sum = 0;
      for (int c = 0; c < classesCount - 1; c++){
        double expX = exp(mxPoint.get(c, i));
        sum += expX;
      }
      final int pointClass = target.at(i);
      for (int c = 0; c < classesCount - 1; c++){
        if (pointClass == c)
          resultMx.adjust(c, i, -(1. + sum - exp(mxPoint.get(c, i)))/(1. + sum));
        else
          resultMx.adjust(c, i, exp(mxPoint.get(c, i))/ (1. + sum));
      }
    }
    return result;
  }

  @Override
  public double value(Vec point) {
    double result = 0;
    Mx mxPoint = new VecBasedMx(target.length(), point);
    for (int i = 0; i < target.length(); i++) {
      double sum = 0;
      for (int c = 0; c < classesCount - 1; c++){
        double expX = exp(mxPoint.get(c, i));
        sum += expX;
      }
      final int pointClass = target.at(i);
      if (pointClass != classesCount - 1)
        result += log(exp(mxPoint.get(pointClass, i)) / (1. + sum));
      else
        result += log(1./(1. + sum));
    }

    return exp(result / target.length());
  }

  @Override
  public int dim() {
    return target.length() * (classesCount - 1);
  }

  public int label(int idx) {
    return target.at(idx);
  }

  public int classesCount() {
    return classesCount;
  }

  public IntSeq labels() {
    return target;
  }
}
