package com.expleague.ml.models.pgm;

import java.util.stream.IntStream;

/**
 * User: solar
 * Date: 07.04.14
 * Time: 21:36
 */
public interface Route {
  double p();
  int last();
  int length();

  ProbabilisticGraphicalModel dstOwner(int stepNo);
  int dst(int stepNo);

  IntStream states();
}
