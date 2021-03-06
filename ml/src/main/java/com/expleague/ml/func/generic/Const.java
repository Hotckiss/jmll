package com.expleague.ml.func.generic;

import org.jetbrains.annotations.NotNull;

/**
 * User: solar
 * Date: 10.06.15
 * Time: 23:40
 */
public class Const extends ElementaryFunc {
  final double value;

  public Const(double value) {
    this.value = value;
  }

  @Override
  public double value(double x) {
    return value;
  }

  @NotNull
  @Override
  public ElementaryFunc gradient() {
    return new Const(0);
  }
}
