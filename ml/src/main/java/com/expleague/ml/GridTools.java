package com.expleague.ml;

import com.expleague.commons.math.AnalyticFunc;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.binarization.partitions.PartitionResult;
import com.expleague.ml.binarization.partitions.PartitionResultBigInt;
import com.expleague.ml.binarization.wrappers.PermutationWrapper;
import com.expleague.ml.binarization.wrappers.SortedFeatureWrapper;
import com.expleague.ml.data.Aggregate;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.ml.impl.BFGridImpl;
import com.expleague.ml.impl.BFRowImpl;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.WeightedLoss;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.TDoubleSet;
import gnu.trove.set.hash.TDoubleHashSet;
import gnu.trove.set.hash.TIntHashSet;

import java.math.BigInteger;
import java.util.*;

import static com.expleague.ml.binarization.calcers.MapperScoreCalcerBigInt.mapperScoreBigInt;
import static com.expleague.ml.binarization.calcers.PartitionScoreCalcers.calculatePartitionScore_hash;
import static com.expleague.ml.binarization.partitions.BestPartitionsSearchers.*;
import static com.expleague.ml.binarization.utils.BinarizationUtils.firstPartition;
import static com.expleague.ml.binarization.utils.BinarizationUtils.insertBorder;
import static com.expleague.ml.binarization.utils.MappersUtils.*;

/**
 * User: solar
 * Date: 27.07.12
 * Time: 17:42
 */
public class GridTools {
  public static TIntArrayList greedyLogSumBorders(final double[] sortedFeature,
                                                  final int binFactor) {
    final TIntArrayList borders = new TIntArrayList();
    borders.add(sortedFeature.length);

    while (borders.size() < binFactor + 1) {
      double bestScore = 0;
      int bestSplit = -1;
      for (int i = 0; i < borders.size(); i++) {
        final int start = i > 0 ? borders.get(i - 1) : 0;
        final int end = borders.get(i);
        final double median = sortedFeature[start + (end - start) / 2];

        int split = Math.abs(Arrays.binarySearch(sortedFeature, start, end, median));

        while (split > 0 && Math.abs(sortedFeature[split] - median) < 1e-9) // look for first less then median value
          split--;
        if (Math.abs(sortedFeature[split] - median) > 1e-9) split++;
        final double scoreLeft = Math.log(end - split) + Math.log(split - start);
        if (split > 0 && scoreLeft > bestScore) {
          bestScore = scoreLeft;
          bestSplit = split;
        }
        while (++split < end && Math.abs(sortedFeature[split] - median) < 1e-9)
          ; // first after elements with such value
        final double scoreRight = Math.log(end - split) + Math.log(split - start);
        if (split < end && scoreRight > bestScore) {
          bestScore = scoreRight;
          bestSplit = split;
        }
      }

      if (bestSplit < 0)
        break;
      borders.add(bestSplit);
      borders.sort();
    }
    return borders;
  }

  public static TDoubleSet uniqueValuesSet(final Vec src) {
    TDoubleHashSet values = new TDoubleHashSet();
    for (int i = 0; i < src.dim(); ++i) {
      values.add(src.get(i));
    }
    return values;
  }

  public static double[] sortUnique(final Vec vec) {
    TDoubleSet values = uniqueValuesSet(vec);
    final double[] result = values.toArray();
    Arrays.sort(result);
    return result;
  }

  public static int uniqueValues(final Vec vec) {
    return uniqueValuesSet(vec).size();
  }

  public static BFGrid medianGrid(final VecDataSet ds, final int binFactor, BuildProgressHandler buildProgressHandler) {
    final int dim = ds.xdim();
    System.out.println("[");
    final BFRowImpl[] rows = new BFRowImpl[dim];
    final TIntHashSet known = new TIntHashSet();
    int bfCount = 0;

    final double[] feature = new double[ds.length()];
    for (int f = 0; f < dim; f++) {
      buildProgressHandler.step();
      final ArrayPermutation permutation = new ArrayPermutation(ds.order(f));
      final int[] order = permutation.direct();
      final int[] reverse = permutation.reverse();

      boolean haveDiffrentElements = false;
      for (int i = 1; i < order.length; i++)
        if (order[i] != order[0])
          haveDiffrentElements = true;
      if (!haveDiffrentElements)
        continue;
      for (int i = 0; i < feature.length; i++) {
        feature[i] = ds.at(order[i]).get(f);
        //printWriter.print(feature[i] + " ");
      }
      //printWriter.println("");


      //sorted feature
      final TIntArrayList borders = greedyLogSumBorders(feature, binFactor);
      final TDoubleArrayList dborders = new TDoubleArrayList();
      final TIntArrayList sizes = new TIntArrayList();
      { // drop existing
        int size = borders.size();
        final int[] crcs = new int[size];
        for (int i = 0; i < ds.length(); i++) { // unordered index
          final int orderedIndex = reverse[i];
          for (int b = 0; b < size && orderedIndex >= borders.get(b); b++) {
            crcs[b] = (crcs[b] * 31) + (i + 1);
          }
        }
        for (int b = 0; b < size - 1; b++) {
          if (known.contains(crcs[b])) {
            //System.out.print("CRCS: " + borders.get(b) + " ");
            continue;
          }
          known.add(crcs[b]);
          int borderValue = borders.get(b);
          dborders.add((feature[borderValue - 1] + feature[borderValue]) / 2.);
          sizes.add(borderValue);
        }
      }
      System.out.print("[");
      StringBuilder sb = new StringBuilder();
      for (int ii = 0; ii < sizes.size(); ii++) {
        sb.append(sizes.get(ii) + ", ");
        //System.out.print(sizes.get(ii) + ", ");
      }
      if (sb.length() > 0) {
        sb.delete(sb.length() - 2, sb.length());
      }

      System.out.println(sb.toString() + "], ");
      //System.out.println("]");
      rows[f] = new BFRowImpl(bfCount, f, dborders.toArray(), sizes.toArray());

      bfCount += dborders.size();
    }
    System.out.println("]");
    System.out.println("BF: " + bfCount);
    //printWriter.close();
    return new BFGridImpl(rows);
  }

  public static class PermutationWeightedFunc extends AnalyticFunc.Stub {
    private final int c;
    private final Aggregate aggregate;
    private final WeightedLoss<? extends L2> loss;
    private final int[] order;

    public PermutationWeightedFunc(int c, int[] order, Aggregate aggregate, WeightedLoss<? extends L2> loss) {
      this.c = c;
      this.order = order;
      this.aggregate = aggregate;
      this.loss = loss;
    }

    @Override
    public double value(double x) {
      double[] params = new double[]{0, 0, 0};
      aggregate.visitND(c, order.length, x, (k, N_k, D_k, P_k, S_k) -> {
        final int index = order[k];
        final double y_k = loss.target().get(index);
        final double w_k = loss.weight(index) * N_k / D_k;

        params[0] += w_k * y_k * y_k;
        params[1] += w_k * y_k;
        params[2] += w_k;
      });
      double sum2 = params[0];
      double sum = params[1];
      double weights = params[2];
      return sum2 - sum * sum / weights;
    }

    @Override
    public double gradient(double x) {
      final double[] params = new double[]{0};
      final WeightedLoss.Stat stat = (WeightedLoss.Stat) aggregate.total();
      final L2.Stat l2Stat = (L2.Stat)stat.inside;
      aggregate.visitND(c, order.length, x, (k, N_k, D_k, P_k, S_k) -> {
        final int index = order[k];
        final double y_k = loss.target().get(index);
        final double w_k = loss.weight(index) * N_k / D_k;

        final double dLdw = y_k * y_k - 2 * (y_k * l2Stat.sum * l2Stat.weight - l2Stat.sum * l2Stat.sum) / l2Stat.weight / l2Stat.weight / l2Stat.weight;
        final double dwdl = (S_k * D_k - P_k * N_k) / N_k / N_k;
        params[0] += w_k * dLdw * dwdl;
      });
      return params[0];
    }
  }
}
