package com.expleague.ml.binarization.utils;

import gnu.trove.list.array.TIntArrayList;

import java.util.HashMap;
import java.util.Map;

import static com.expleague.ml.binarization.utils.BinarizationUtils.insertBorder;

public class MappersUtils {
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
     * Calculates score for current binarization mapper
     * @param mapper
     * @return
     */
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
     * Prints current mapper
     * @param mapper
     */
    public static void outMapper(HashMap<Integer, HashMap<Integer, Integer>> mapper) {
        System.out.println("\n---Mapper---");
        for (Map.Entry<Integer, HashMap<Integer, Integer>> entry : mapper.entrySet()) {
            System.out.println("F1_bin: " + entry.getKey());
            for (Map.Entry<Integer, Integer> entry1 : entry.getValue().entrySet()) {
                System.out.println("    F2_bin: " + entry1.getKey() + " Cnt: " + entry1.getValue());
            }
        }
        System.out.println("------");
    }

    /**
     * Builds counters mapper for suggested partitions
     * @param binNumberMapper
     * @param sortedFeature2
     * @param bordersFeature2
     * @param newBorderFeature2
     * @return
     */
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
}
