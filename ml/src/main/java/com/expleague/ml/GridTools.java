package com.expleague.ml;

import com.expleague.commons.math.AnalyticFunc;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.util.ArrayTools;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * User: solar
 * Date: 27.07.12
 * Time: 17:42
 */
public class GridTools {
  public static class PartitionResult {
    private int splitPosition;
    private double score;

    public PartitionResult(int splitPosition, double score) {
      this.score = score;
      this.splitPosition = splitPosition;
    }

    public int getSplitPosition() {
      return splitPosition;
    }

    public double getScore() {
      return score;
    }

    public void setSplitPosition(int newSplitPosition) {
      this.splitPosition = newSplitPosition;
    }

    public void setScore(double newScore) {
      this.score = newScore;
    }

    public static PartitionResult makeWorst() {
      return new PartitionResult(-1, -Double.MAX_VALUE);
    }
  }

  public static TIntArrayList insertBorder(final TIntArrayList borders, final int newBorder) {
    final TIntArrayList mergedBorders = new TIntArrayList();
    int i = 0;

    while (i < borders.size() && borders.get(i) < newBorder) {
      mergedBorders.add(borders.get(i));
      i++;
    }

    mergedBorders.add(newBorder);

    while (i < borders.size() && borders.get(i) == newBorder) {
      i++;
    }

    while (i < borders.size()) {
      mergedBorders.add(borders.get(i));
      i++;
    }

    return mergedBorders;
  }

  /**
   * Builds map: take value in position number i in sorted order by feature2 then
   * return number of bin of this element in feature1 binarization
   * All elements are in 0-ind
   * Complexity: O(n) [3 * n]
   * @param bordersFeature2 bins borders of feature 1 in sorted order
   * @param sortedOrderIndicesFeature2 indices of sorted order for feature2
   * @param sortedOrderIndicesFeature1 indices of sorted order for feature1
   * @return result mapper
   */
  public static int[] buildBinsMapper(final TIntArrayList bordersFeature2,
                                      final int[] sortedOrderIndicesFeature2,
                                      final int[] sortedOrderIndicesFeature1) {

    //build reverse permutation to easy search from feature2
    int[] feature2Mapper = new int[sortedOrderIndicesFeature2.length];
    for (int i = 0; i < sortedOrderIndicesFeature2.length; i++) {
      feature2Mapper[sortedOrderIndicesFeature2[i]] = i;
    }

    //build bins number mapper
    int[] binNumberMapper = new int[sortedOrderIndicesFeature2.length];
    int binNumber = 0;
    for (int i = 0; i < bordersFeature2.size(); i++) {
      final int start = i > 0 ? bordersFeature2.get(i - 1) : 0;
      final int end = bordersFeature2.get(i);
      for (int j = start; j < end; j++) {
        binNumberMapper[j] = binNumber;
      }

      binNumber++;
    }

    int[] resultMapper = new int[sortedOrderIndicesFeature1.length];
    for (int i = 0; i < sortedOrderIndicesFeature1.length; i++) {
      int positionInFeature1Order = feature2Mapper[sortedOrderIndicesFeature1[i]];
      resultMapper[i] = binNumberMapper[positionInFeature1Order];
    }

    return resultMapper;
  }

  /**
   * Given two sorted features with existing borders
   * We want to add new border in feature2 and we want to calculate quality of this partition
   * Quality calculation looks like this:
   * 1) Iterate over each point in dataset and determine it's bin in feature1
   * 2) Now we want to calculate the probability of choosing correct position in order by feature2 for each point
   * To do that, we iterate through every bin in second feature, and for each point determine it's bin from feature1
   * Probability will be 1/number of points from the same bin in feature1 as this point
   * Quality will be sum for each point log(probability)
   * Complexity: O(n) [2 * n]
   * @param binNumberMapper for point i of sortedFeature2 determines it's bin in feature1
   * @param sortedFeature2 points, sorted in ascending order from feature2
   * @param bordersFeature2 current fixed borders of feature2, must contain at least sortedFeature2.length
   * @param newBorderFeature2 new possible border of feature2
   * @return quality score of new border using input feature1
   */
  //TODO: instead of max sum log(1/n) = max -sum log(n) = max - log(prod n) ~ min log(prod n) = min prod n
  public static double calculatePartitionScore(final int[] binNumberMapper,
                                               final double[] sortedFeature2,
                                               final TIntArrayList bordersFeature2,
                                               final int newBorderFeature2) {
    // O(binFactor)
    TIntArrayList mergedBordersFeature2 = insertBorder(bordersFeature2, newBorderFeature2);

    double score = 0.0;

    // i -- number of right border of bin
    //O(n)
    //System.out.println("Border: " + (newBorderFeature2 + 1));
    //System.out.print("Probabilities: ");
    for (int i = 0; i < mergedBordersFeature2.size(); i++) {
      final int start = i > 0 ? mergedBordersFeature2.get(i - 1) : 0;
      final int end = mergedBordersFeature2.get(i);

      //for points from this bin calculate number of points in each bin in feature1
      int[] binsCounters = new int[binNumberMapper.length];
      for (int position = start; position < end; position++) {
        int binInFeature1 = binNumberMapper[position];
        binsCounters[binInFeature1]++;
      }

      //calculate probabilities
      for (int position = start; position < end; position++) {
        int binInFeature1 = binNumberMapper[position];
        //System.out.print((1.0 / binsCounters[binInFeature1]) + " ");
        score += Math.log(1.0 / (binsCounters[binInFeature1] + 1));
      }

    }
    //System.out.println();
    return score;
  }

  public static double calculatePartitionScore_hash(final int[] binNumberMapper,
                                               final double[] sortedFeature2,
                                               final TIntArrayList bordersFeature2,
                                               final int newBorderFeature2) {
    // O(binFactor)
    TIntArrayList mergedBordersFeature2 = insertBorder(bordersFeature2, newBorderFeature2);

    double score = 0.0;

    // i -- number of right border of bin
    //O(n)
    //System.out.println("Border: " + (newBorderFeature2 + 1));
    //System.out.print("Probabilities: ");
    for (int i = 0; i < mergedBordersFeature2.size(); i++) {
      final int start = i > 0 ? mergedBordersFeature2.get(i - 1) : 0;
      final int end = mergedBordersFeature2.get(i);

      //for points from this bin calculate number of points in each bin in feature1
      HashMap<Integer, Integer> binsCountersHash = new HashMap<>();
      for (int position = start; position < end; position++) {
        int binInFeature1 = binNumberMapper[position];
        Integer cur_val = binsCountersHash.get(binInFeature1);
        binsCountersHash.put(binInFeature1, (cur_val == null ? 1 : cur_val + 1));
      }

      //calculate probabilities
      for (Map.Entry<Integer, Integer> entry : binsCountersHash.entrySet()) {
        score += entry.getValue() * Math.log(1.0 / (entry.getValue() + 1));
      }
    }

    return score;
  }

  public static HashMap<Integer, HashMap<Integer, Integer>> partitionCountersMapper(final int[] binNumberMapper,
                                                    final double[] sortedFeature2,
                                                    final TIntArrayList bordersFeature2,
                                                    final int newBorderFeature2) {
    TIntArrayList mergedBordersFeature2 = insertBorder(bordersFeature2, newBorderFeature2);

    //mapper bin in feature2->hash map with counters for each element
    HashMap<Integer, HashMap<Integer, Integer>> mapper = new HashMap<>();

    for (int i = 0; i < mergedBordersFeature2.size(); i++) {
      final int start = i > 0 ? mergedBordersFeature2.get(i - 1) : 0;
      final int end = mergedBordersFeature2.get(i);
      HashMap<Integer, Integer> binsCountersHash = new HashMap<>();

      for (int position = start; position < end; position++) {
        int binInFeature1 = binNumberMapper[position];
        Integer cur_val = binsCountersHash.get(binInFeature1);
        binsCountersHash.put(binInFeature1, (cur_val == null ? 1 : cur_val + 1));
      }

      mapper.put(i, binsCountersHash);
    }

    return mapper;
  }

  public static double mapperScore(HashMap<Integer, HashMap<Integer, Integer>> mapper) {
    double score = 0.0;
    for (Map.Entry<Integer, HashMap<Integer, Integer>> entry : mapper.entrySet()) {
      for (Map.Entry<Integer, Integer> entry1 : entry.getValue().entrySet()) {
        score += entry1.getValue() * Math.log(1.0 / (entry1.getValue() + 1));
      }
    }

    return score;
  }

  /**
   * Finds the best partition of feature2 according to feature1 and current binarization
   * Complexity: O(n) worst-case [2 * n * |bin|]
   * @param binNumberMapper
   * @param sortedFeature2
   * @param bordersFeature2
   * @return best border and score
   */
  public static PartitionResult bestPartitionWithMapper(final int[] binNumberMapper,
                                              final double[] sortedFeature2,
                                              final TIntArrayList bordersFeature2) {
    int startPivot = firstPartition(bordersFeature2);
    //System.out.println("First pivot: " + startPivot);
    int bordersPtr = 0;
    HashMap<Integer, HashMap<Integer, Integer>> currentMapper = partitionCountersMapper(binNumberMapper, sortedFeature2, bordersFeature2, startPivot);
    PartitionResult bestRes = new PartitionResult(startPivot, mapperScore(currentMapper));
    PartitionResult lastRes = new PartitionResult(startPivot, mapperScore(currentMapper));

    for (int pivot = startPivot + 1; pivot < binNumberMapper.length; pivot++) {
      { // check that border doesn't exist
        while (bordersPtr < bordersFeature2.size() && bordersFeature2.get(bordersPtr) < pivot) {
          bordersPtr++;
        }
        if (bordersPtr < bordersFeature2.size() && bordersFeature2.get(bordersPtr) == pivot) {
          continue;
        }
      }
      //System.out.println("Pivot: " + pivot);
      // only one element has change bin number in second feature
      if (pivot == lastRes.getSplitPosition() + 1) {

        int movedElementBinInFeature1 = binNumberMapper[pivot - 1];
        int movedElementBinInFeature2 = bordersPtr + 1;
        //System.out.println("F1: " + movedElementBinInFeature1 + " F2: " + movedElementBinInFeature2);
        //System.out.println("Keys: " + currentMapper.keySet());
        //decrement for old bin
        int value_old = currentMapper.get(movedElementBinInFeature2).get(movedElementBinInFeature1);
        if (value_old == 1) {
          currentMapper.get(movedElementBinInFeature2).remove(movedElementBinInFeature1);
        } else {
          currentMapper.get(movedElementBinInFeature2).put(movedElementBinInFeature1, value_old - 1);
        }

        //increment for new bin
        Integer value_old1 = currentMapper.get(movedElementBinInFeature2 - 1).get(movedElementBinInFeature1);
        currentMapper.get(movedElementBinInFeature2 - 1).put(movedElementBinInFeature1, (value_old1 == null ? 1 : value_old1 + 1));

        double score = mapperScore(currentMapper);

        if (Math.abs(sortedFeature2[pivot] - sortedFeature2[pivot - 1]) < 1e-9) {
          lastRes = new PartitionResult(pivot, score);
          continue;
        }

        if (score > bestRes.score) {
          bestRes.setScore(score);
          bestRes.setSplitPosition(pivot);
        }

        lastRes = new PartitionResult(pivot, score);
      } else {
        currentMapper = partitionCountersMapper(binNumberMapper, sortedFeature2, bordersFeature2, pivot);
        double score = mapperScore(currentMapper);

        if (Math.abs(sortedFeature2[pivot] - sortedFeature2[pivot - 1]) < 1e-9) {
          lastRes = new PartitionResult(pivot, score);
          continue;
        }

        if (score > bestRes.score) {
          bestRes.setScore(score);
          bestRes.setSplitPosition(pivot);
        }

        lastRes = new PartitionResult(pivot, score);
      }
    }

    return bestRes;
  }

  public static int firstPartition(final TIntArrayList bordersFeature2) {
    int res = 1;
    int bordersPtr = 0;
    while (bordersPtr < bordersFeature2.size() && bordersFeature2.get(bordersPtr) == res) {
      bordersPtr++;
      res++;
    }

    return res;
  }

  public static PartitionResult bestPartitionWithMapper2(final int[] binNumberMapper,
                                                        final double[] sortedFeature2,
                                                        final TIntArrayList bordersFeature2) {
    int startPivot = firstPartition(bordersFeature2);
    int bordersPtr = 0;

    HashMap<Integer, HashMap<Integer, Integer>> currentMapper = partitionCountersMapper(binNumberMapper, sortedFeature2, bordersFeature2, startPivot);
    PartitionResult bestRes = new PartitionResult(startPivot, mapperScore(currentMapper));
    PartitionResult lastRes = new PartitionResult(startPivot, mapperScore(currentMapper));


    for (int pivot = startPivot + 1; pivot < binNumberMapper.length; pivot++) {
      { // check that border doesn't exist
        while (bordersPtr < bordersFeature2.size() && bordersFeature2.get(bordersPtr) < pivot) {
          bordersPtr++;
        }
        if (bordersPtr < bordersFeature2.size() && bordersFeature2.get(bordersPtr) == pivot) {
          continue;
        }
      }

      currentMapper = partitionCountersMapper(binNumberMapper, sortedFeature2, bordersFeature2, pivot);
      double score = mapperScore(currentMapper);

      if (score > bestRes.score) {
        bestRes.setScore(score);
        bestRes.setSplitPosition(pivot);
      }

      lastRes = new PartitionResult(pivot, score);
    }

    return bestRes;
  }

  /**
   * Finds the best partition of feature2 according to feature1 and current binarization
   * Complexity: O(n^2) worst-case [2 * n^2]
   * @param binNumberMapper
   * @param sortedFeature2
   * @param bordersFeature2
   * @return best border and score
   */
  public static PartitionResult bestPartition(final int[] binNumberMapper,
                                  final double[] sortedFeature2,
                                  final TIntArrayList bordersFeature2) {
    double bestScore = -Double.MAX_VALUE;
    int bestSplit = -1;
    int bordersPtr = 0;
    for (int pivot = 1; pivot < binNumberMapper.length; pivot++) {
      // check that border doesn't exist
      {
        while (bordersPtr < bordersFeature2.size() && bordersFeature2.get(bordersPtr) < pivot) {
          bordersPtr++;
        }
        if (bordersPtr < bordersFeature2.size() && bordersFeature2.get(bordersPtr) == pivot) {
          continue;
        }
      }

      double score = calculatePartitionScore_hash(binNumberMapper, sortedFeature2, bordersFeature2, pivot);
      // maximize sum log(1/n), revert after TODO
      //System.out.println("Score: " + score);
      //System.out.println();
      if (score > bestScore) {
        bestScore = score;
        bestSplit = pivot;
      }
    }

    //System.out.println("------------------");
    return new PartitionResult(bestSplit, bestScore);
  }

  /**
   * Builds naive probability grid
   * @param ds
   * @param binFactor
   * @param useFastAlgorithm using fast algorithm for partition search
   * @return
   */
  public static BFGrid probabilityGrid(final VecDataSet ds, final int binFactor, boolean useFastAlgorithm) {
    assert (binFactor < ds.length());

    final int dim = ds.xdim();
    final BFRowImpl[] rows = new BFRowImpl[dim];
    int bfCount = 0;
    ArrayList<TIntArrayList> currentBorders = new ArrayList<>();

    //initial borders
    for (int i = 0; i < dim; i++) {
      TIntArrayList borders = new TIntArrayList();
      borders.add(ds.length());
      currentBorders.add(borders);
    }

    for(int iters = 0; iters < binFactor; iters++) {
      System.out.println("Iter: " + iters);
      for (int feature_index = 0; feature_index < dim; feature_index++) {
        System.out.print("Feature: " + feature_index + " ");
        final double[] feature = new double[ds.length()];
        final ArrayPermutation permutation = new ArrayPermutation(ds.order(feature_index));
        final int[] order = permutation.direct();
        final int[] reverse = permutation.reverse();
        boolean haveDiffrentElements = false;
        for (int i = 1; i < order.length; i++)
          if (order[i] != order[0])
            haveDiffrentElements = true;
        if (!haveDiffrentElements)
          continue;

        for (int i = 0; i < feature.length; i++)
          feature[i] = ds.at(order[i]).get(feature_index);

        PartitionResult bestFromAll = PartitionResult.makeWorst();

        int bf = 0;
        for (int paired_feature_index = 0; paired_feature_index < dim; paired_feature_index++) {
          //System.out.println(paired_feature_index);
          if (paired_feature_index == feature_index) {
            continue;
          }

          final ArrayPermutation permutationPaired = new ArrayPermutation(ds.order(paired_feature_index));
          final int[] reversePaired = permutationPaired.reverse();

          int[] binNumberMapper = buildBinsMapper(currentBorders.get(paired_feature_index), reverse, reversePaired);

          PartitionResult bestResult = PartitionResult.makeWorst();

          if (useFastAlgorithm) {
            bestResult = bestPartitionWithMapper(binNumberMapper, feature, currentBorders.get(feature_index));

          } else {
            bestResult = bestPartition(binNumberMapper, feature, currentBorders.get(feature_index));
          }

          if (bestFromAll.score < bestResult.score) {
            bestFromAll = bestResult;
            bf = paired_feature_index;
          }
        }

        //System.out.println(bestFromAll.splitPosition);
        if (bestFromAll.splitPosition > 1) {
          System.out.println("BestPaired feature: " + bf);
          TIntArrayList newBorders = insertBorder(currentBorders.get(feature_index), bestFromAll.splitPosition);
          currentBorders.set(feature_index, newBorders);
        } else {
          System.out.println();
        }

      }
    }

    for (int i = 0; i < currentBorders.size(); i++) {
      for (int j = 0; j < currentBorders.get(i).size() - 1; j++) {
        System.out.print(currentBorders.get(i).get(j) + " ");
      }
      System.out.println();
    }

    for (int f = 0; f < dim; f++) {
      final TIntArrayList borders = currentBorders.get(f);
      int size = borders.size();
      final TDoubleArrayList dborders = new TDoubleArrayList();
      final TIntArrayList sizes = new TIntArrayList();
      final double[] feature = new double[ds.length()];
      final ArrayPermutation permutation = new ArrayPermutation(ds.order(f));
      final int[] order = permutation.direct();
      for (int i = 0; i < feature.length; i++)
        feature[i] = ds.at(order[i]).get(f);

      for (int b = 0; b < size - 1; b++) {
        int borderValue = borders.get(b);
        dborders.add((feature[borderValue - 1] + feature[borderValue]) / 2.);
        sizes.add(borderValue);
      }
      rows[f] = new BFRowImpl(bfCount, f, dborders.toArray(), sizes.toArray());
      bfCount += dborders.size();
    }

    return new BFGridImpl(rows);
  }

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

  public static BFGrid medianGrid(final VecDataSet ds, final int binFactor) {
    final int dim = ds.xdim();
    final BFRowImpl[] rows = new BFRowImpl[dim];
    final TIntHashSet known = new TIntHashSet();
    int bfCount = 0;

    final double[] feature = new double[ds.length()];
    for (int f = 0; f < dim; f++) {
      final ArrayPermutation permutation = new ArrayPermutation(ds.order(f));
      final int[] order = permutation.direct();
      final int[] reverse = permutation.reverse();

      boolean haveDiffrentElements = false;
      for (int i = 1; i < order.length; i++)
        if (order[i] != order[0])
          haveDiffrentElements = true;
      if (!haveDiffrentElements)
        continue;
      for (int i = 0; i < feature.length; i++)
        feature[i] = ds.at(order[i]).get(f);

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
      for (int ii = 0; ii < sizes.size(); ii++) {
        System.out.print(sizes.get(ii) + " ");
      }
      System.out.println();
      rows[f] = new BFRowImpl(bfCount, f, dborders.toArray(), sizes.toArray());

      bfCount += dborders.size();
    }
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
