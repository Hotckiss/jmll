package com.expleague.ml.data.stats;

import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.data.set.VecDataSet;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.function.Function;

/**
 * Created with IntelliJ IDEA.
 * User: solar
 * Date: 22.03.13
 * Time: 20:32
 * To change this template use File | Settings | File Templates.
 */
public class OrderByFeature implements Function<DataSet, OrderByFeature> {
  final TIntObjectHashMap<ArrayPermutation> orders = new TIntObjectHashMap<ArrayPermutation>();
  VecDataSet set;

  @Override
  public OrderByFeature apply(final DataSet argument) {
    set = (VecDataSet)argument;
    return this;
  }

  public synchronized ArrayPermutation orderBy(final int featureNo) {
    ArrayPermutation result = orders.get(featureNo);
    if (result == null)
      orders.put(featureNo, result = new ArrayPermutation(set.order(featureNo)));
    return result;
  }
}
