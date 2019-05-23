package com.expleague.ml.binarization.calcers;

import gnu.trove.list.array.TIntArrayList;

import java.util.HashMap;
import java.util.Map;

import static com.expleague.ml.binarization.utils.BinarizationUtils.insertBorder;

public class PartitionScoreCalcers {
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

    /**
     * Same calser as above, but using hash map for counters. Two times faster
     * @param binNumberMapper
     * @param sortedFeature2
     * @param bordersFeature2
     * @param newBorderFeature2
     * @return
     */
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
}
