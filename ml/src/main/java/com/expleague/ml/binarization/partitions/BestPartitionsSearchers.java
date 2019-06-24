package com.expleague.ml.binarization.partitions;

import gnu.trove.list.array.TIntArrayList;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static com.expleague.ml.binarization.calcers.MapperScoreCalcerBigInt.mapperScoreBigInt;
import static com.expleague.ml.binarization.calcers.PartitionScoreCalcers.calculatePartitionScore_hash;
import static com.expleague.ml.binarization.utils.BinarizationUtils.firstPartition;
import static com.expleague.ml.binarization.utils.MappersUtils.mapperScore;
import static com.expleague.ml.binarization.utils.MappersUtils.partitionCountersMapper;

public class BestPartitionsSearchers {
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

            // only one element has change bin number in second feature
            if (pivot == lastRes.getSplitPosition() + 1) {

                int movedElementBinInFeature1 = binNumberMapper[pivot - 1];
                int movedElementBinInFeature2 = bordersPtr + 1;

                //decrement for old bin
                int value_old = currentMapper.get(movedElementBinInFeature2).get(movedElementBinInFeature1);
                double contribution_old1 = value_old * Math.log(1.0 / (value_old + 1));
                double contribution_new1 = (value_old - 1) * Math.log(1.0 / (value_old));
                if (value_old == 1) {
                    currentMapper.get(movedElementBinInFeature2).remove(movedElementBinInFeature1);
                } else {
                    currentMapper.get(movedElementBinInFeature2).put(movedElementBinInFeature1, value_old - 1);
                }

                //increment for new bin
                Integer value_old1 = currentMapper.get(movedElementBinInFeature2 - 1).get(movedElementBinInFeature1);
                value_old1 = (value_old1 == null ? 0 : value_old1);
                double contribution_old2 = value_old1 * Math.log(1.0 / (value_old1 + 1));
                double contribution_new2 = (value_old1 + 1) * Math.log(1.0 / (value_old1 + 2));
                currentMapper.get(movedElementBinInFeature2 - 1).put(movedElementBinInFeature1, (value_old1 + 1));


                double score = lastRes.getScore() - contribution_old1 - contribution_old2 + contribution_new1 + contribution_new2;

                if (Math.abs(sortedFeature2[pivot] - sortedFeature2[pivot - 1]) < 1e-9) {
                    lastRes = new PartitionResult(pivot, score);
                    continue;
                }

                if (score > bestRes.getScore()) {
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

                if (score > bestRes.getScore()) {
                    bestRes.setScore(score);
                    bestRes.setSplitPosition(pivot);
                }

                lastRes = new PartitionResult(pivot, score);
            }
        }

        return bestRes;
    }

    /**
     * Fast moving through existing border without rebuild
     * @param binNumberMapper
     * @param sortedFeature2
     * @param bordersFeature2
     * @return
     */
    public static PartitionResult bestPartitionWithMapper_veryFast(final int[] binNumberMapper,
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

            // only one element has change bin number in second feature
            if (pivot == lastRes.getSplitPosition() + 1) {

                int movedElementBinInFeature1 = binNumberMapper[pivot - 1];
                int movedElementBinInFeature2 = bordersPtr + 1;

                //decrement for old bin
                int value_old = currentMapper.get(movedElementBinInFeature2).get(movedElementBinInFeature1);
                double contribution_old1 = value_old * Math.log(1.0 / (value_old + 1));
                double contribution_new1 = (value_old - 1) * Math.log(1.0 / (value_old));
                if (value_old == 1) {
                    currentMapper.get(movedElementBinInFeature2).remove(movedElementBinInFeature1);
                } else {
                    currentMapper.get(movedElementBinInFeature2).put(movedElementBinInFeature1, value_old - 1);
                }

                //increment for new bin
                Integer value_old1 = currentMapper.get(movedElementBinInFeature2 - 1).get(movedElementBinInFeature1);
                value_old1 = (value_old1 == null ? 0 : value_old1);
                double contribution_old2 = value_old1 * Math.log(1.0 / (value_old1 + 1));
                double contribution_new2 = (value_old1 + 1) * Math.log(1.0 / (value_old1 + 2));
                currentMapper.get(movedElementBinInFeature2 - 1).put(movedElementBinInFeature1, (value_old1 + 1));

                double score = lastRes.getScore() - contribution_old1 - contribution_old2 + contribution_new1 + contribution_new2;

                if (Math.abs(sortedFeature2[pivot] - sortedFeature2[pivot - 1]) < 1e-9) {
                    lastRes = new PartitionResult(pivot, score);
                    continue;
                }

                if (score > bestRes.getScore()) {
                    bestRes.setScore(score);
                    bestRes.setSplitPosition(pivot);
                }

                lastRes = new PartitionResult(pivot, score);
            } else if (pivot == lastRes.getSplitPosition() + 2) {
                //old bins numbers in feature2
                int movedElement1_binInFeature2 = bordersPtr;
                int movedElement2_binInFeature2 = bordersPtr + 1;

                //old bins numbers in feature1
                int movedElement1_BinInFeature1 = binNumberMapper[pivot - 2];
                int movedElement2_BinInFeature1 = binNumberMapper[pivot - 1];

                // old values and contributions
                int value_old_el1 = currentMapper.get(movedElement1_binInFeature2).get(movedElement1_BinInFeature1);
                double contribution_old1_el1 = value_old_el1 * Math.log(1.0 / (value_old_el1 + 1));
                double contribution_new1_el1 = (value_old_el1 - 1) * Math.log(1.0 / (value_old_el1));
                if (value_old_el1 == 1) {
                    currentMapper.get(movedElement1_binInFeature2).remove(movedElement1_BinInFeature1);
                } else {
                    currentMapper.get(movedElement1_binInFeature2).put(movedElement1_BinInFeature1, value_old_el1 - 1);
                }

                int value_old_el2 = currentMapper.get(movedElement2_binInFeature2).get(movedElement2_BinInFeature1);
                double contribution_old1_el2 = value_old_el2 * Math.log(1.0 / (value_old_el2 + 1));
                double contribution_new1_el2 = (value_old_el2 - 1) * Math.log(1.0 / (value_old_el2));
                if (value_old_el2 == 1) {
                    currentMapper.get(movedElement2_binInFeature2).remove(movedElement2_BinInFeature1);
                } else {
                    currentMapper.get(movedElement2_binInFeature2).put(movedElement2_BinInFeature1, value_old_el2 - 1);
                }

                //new values and contributions
                Integer value_old1_el1 = currentMapper.get(movedElement1_binInFeature2 - 1).get(movedElement1_BinInFeature1);
                value_old1_el1 = (value_old1_el1 == null ? 0 : value_old1_el1);
                double contribution_old2_el1 = value_old1_el1 * Math.log(1.0 / (value_old1_el1 + 1));
                double contribution_new2_el1 = (value_old1_el1 + 1) * Math.log(1.0 / (value_old1_el1 + 2));
                currentMapper.get(movedElement1_binInFeature2 - 1).put(movedElement1_BinInFeature1, (value_old1_el1 + 1));

                Integer value_old1_el2 = currentMapper.get(movedElement2_binInFeature2 - 1).get(movedElement2_BinInFeature1);
                value_old1_el2 = (value_old1_el2 == null ? 0 : value_old1_el2);
                double contribution_old2_el2 = value_old1_el2 * Math.log(1.0 / (value_old1_el2 + 1));
                double contribution_new2_el2 = (value_old1_el2 + 1) * Math.log(1.0 / (value_old1_el2 + 2));
                currentMapper.get(movedElement2_binInFeature2 - 1).put(movedElement2_BinInFeature1, (value_old1_el2 + 1));

                double score = lastRes.getScore() - contribution_old1_el1 - contribution_old1_el2 -
                        contribution_old2_el1 - contribution_old2_el2 +
                        contribution_new1_el1 + contribution_new1_el2 +
                        contribution_new2_el1 + contribution_new2_el2;

                if (Math.abs(sortedFeature2[pivot] - sortedFeature2[pivot - 1]) < 1e-9) {
                    lastRes = new PartitionResult(pivot, score);
                    continue;
                }

                if (score > bestRes.getScore()) {
                    bestRes.setScore(score);
                    bestRes.setSplitPosition(pivot);
                }

                lastRes = new PartitionResult(pivot, score);
            } else {
                System.err.println("Rebuild mapper.\nThis means that exists two borders next to each other.\nThis will decrease performance and speed.\nPlease, make binFactor smaller."); // DENSE partition, signal to decrease binFactor
                currentMapper = partitionCountersMapper(binNumberMapper, sortedFeature2, bordersFeature2, pivot);
                double score = mapperScore(currentMapper);

                if (Math.abs(sortedFeature2[pivot] - sortedFeature2[pivot - 1]) < 1e-9) {
                    lastRes = new PartitionResult(pivot, score);
                    continue;
                }

                if (score > bestRes.getScore()) {
                    bestRes.setScore(score);
                    bestRes.setSplitPosition(pivot);
                }

                lastRes = new PartitionResult(pivot, score);
            }
        }

        return bestRes;
    }

    /**
     * Choose best using median
     * @param binNumberMapper
     * @param sortedFeature2
     * @param bordersFeature2
     * @return
     */
    public static PartitionResult bestPartitionWithMapper_veryFast_medianBest(final int[] binNumberMapper,
                                                                              final double[] sortedFeature2,
                                                                              final TIntArrayList bordersFeature2) {
        int startPivot = firstPartition(bordersFeature2);
        //System.out.println("First pivot: " + startPivot);
        int bordersPtr = 0;
        HashMap<Integer, HashMap<Integer, Integer>> currentMapper = partitionCountersMapper(binNumberMapper, sortedFeature2, bordersFeature2, startPivot);
        PartitionResult bestRes = new PartitionResult(startPivot, mapperScore(currentMapper));
        List<PartitionResult> bestResults = new ArrayList<>();
        bestResults.add(bestRes);
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
                double contribution_old1 = value_old * Math.log(1.0 / (value_old + 1));
                double contribution_new1 = (value_old - 1) * Math.log(1.0 / (value_old));
                if (value_old == 1) {
                    currentMapper.get(movedElementBinInFeature2).remove(movedElementBinInFeature1);
                } else {
                    currentMapper.get(movedElementBinInFeature2).put(movedElementBinInFeature1, value_old - 1);
                }

                //increment for new bin
                Integer value_old1 = currentMapper.get(movedElementBinInFeature2 - 1).get(movedElementBinInFeature1);
                value_old1 = (value_old1 == null ? 0 : value_old1);
                double contribution_old2 = value_old1 * Math.log(1.0 / (value_old1 + 1));
                double contribution_new2 = (value_old1 + 1) * Math.log(1.0 / (value_old1 + 2));
                currentMapper.get(movedElementBinInFeature2 - 1).put(movedElementBinInFeature1, (value_old1 + 1));

                //double score = mapperScore(currentMapper);

                double score = lastRes.getScore() - contribution_old1 - contribution_old2 + contribution_new1 + contribution_new2;

                if (Math.abs(sortedFeature2[pivot] - sortedFeature2[pivot - 1]) < 1e-9) {
                    lastRes = new PartitionResult(pivot, score);
                    continue;
                }

                if (Math.abs(score - bestRes.getScore()) < 0.1) {
                    bestResults.add(new PartitionResult(pivot, score));
                }

                if (score > bestRes.getScore()) {
                    if (Math.abs(score - bestRes.getScore()) > 0.1) {
                        bestResults = new ArrayList<>();
                        bestResults.add(new PartitionResult(pivot, score));
                    }
                    bestRes.setScore(score);
                    bestRes.setSplitPosition(pivot);
                }

                lastRes = new PartitionResult(pivot, score);
            } else if (pivot == lastRes.getSplitPosition() + 2) {
                //outMapper(currentMapper);
                //old bins numbers in feature2
                int movedElement1_binInFeature2 = bordersPtr;
                int movedElement2_binInFeature2 = bordersPtr + 1;

                //old bins numbers in feature1
                int movedElement1_BinInFeature1 = binNumberMapper[pivot - 2];
                int movedElement2_BinInFeature1 = binNumberMapper[pivot - 1];

                // old values and contributions

                int value_old_el1 = currentMapper.get(movedElement1_binInFeature2).get(movedElement1_BinInFeature1);
                double contribution_old1_el1 = value_old_el1 * Math.log(1.0 / (value_old_el1 + 1));
                double contribution_new1_el1 = (value_old_el1 - 1) * Math.log(1.0 / (value_old_el1));
                if (value_old_el1 == 1) {
                    currentMapper.get(movedElement1_binInFeature2).remove(movedElement1_BinInFeature1);
                } else {
                    currentMapper.get(movedElement1_binInFeature2).put(movedElement1_BinInFeature1, value_old_el1 - 1);
                }

                int value_old_el2 = currentMapper.get(movedElement2_binInFeature2).get(movedElement2_BinInFeature1);
                double contribution_old1_el2 = value_old_el2 * Math.log(1.0 / (value_old_el2 + 1));
                double contribution_new1_el2 = (value_old_el2 - 1) * Math.log(1.0 / (value_old_el2));
                if (value_old_el2 == 1) {
                    currentMapper.get(movedElement2_binInFeature2).remove(movedElement2_BinInFeature1);
                } else {
                    currentMapper.get(movedElement2_binInFeature2).put(movedElement2_BinInFeature1, value_old_el2 - 1);
                }

                //new values and contributions
                Integer value_old1_el1 = currentMapper.get(movedElement1_binInFeature2 - 1).get(movedElement1_BinInFeature1);
                value_old1_el1 = (value_old1_el1 == null ? 0 : value_old1_el1);
                double contribution_old2_el1 = value_old1_el1 * Math.log(1.0 / (value_old1_el1 + 1));
                double contribution_new2_el1 = (value_old1_el1 + 1) * Math.log(1.0 / (value_old1_el1 + 2));
                currentMapper.get(movedElement1_binInFeature2 - 1).put(movedElement1_BinInFeature1, (value_old1_el1 + 1));

                Integer value_old1_el2 = currentMapper.get(movedElement2_binInFeature2 - 1).get(movedElement2_BinInFeature1);
                value_old1_el2 = (value_old1_el2 == null ? 0 : value_old1_el2);
                double contribution_old2_el2 = value_old1_el2 * Math.log(1.0 / (value_old1_el2 + 1));
                double contribution_new2_el2 = (value_old1_el2 + 1) * Math.log(1.0 / (value_old1_el2 + 2));
                currentMapper.get(movedElement2_binInFeature2 - 1).put(movedElement2_BinInFeature1, (value_old1_el2 + 1));

                //System.out.println("OPTIMIZED");
                //System.out.print("----------------------updated----------------------");
                //outMapper(currentMapper);
                //double score2 = mapperScore(currentMapper);

                double score = lastRes.getScore() - contribution_old1_el1 - contribution_old1_el2 -
                        contribution_old2_el1 - contribution_old2_el2 +
                        contribution_new1_el1 + contribution_new1_el2 +
                        contribution_new2_el1 + contribution_new2_el2;

                //double sc2 = mapperScore(partitionCountersMapper(binNumberMapper, sortedFeature2, bordersFeature2, pivot));

                //System.out.println("OldSc: " + sc2 + " Nnew: " + score + " Add: " + score2);

                if (Math.abs(sortedFeature2[pivot] - sortedFeature2[pivot - 1]) < 1e-9) {
                    lastRes = new PartitionResult(pivot, score);
                    continue;
                }

                if (Math.abs(score - bestRes.getScore()) < 0.1) {
                    bestResults.add(new PartitionResult(pivot, score));
                }

                if (score > bestRes.getScore()) {
                    if (Math.abs(score - bestRes.getScore()) > 0.1) {
                        bestResults = new ArrayList<>();
                        bestResults.add(new PartitionResult(pivot, score));
                    }
                    bestRes.setScore(score);
                    bestRes.setSplitPosition(pivot);
                }

                lastRes = new PartitionResult(pivot, score);

            } else {
                System.out.println("REBUILD"); // DENSE partition, signal to decrease binFactor
                currentMapper = partitionCountersMapper(binNumberMapper, sortedFeature2, bordersFeature2, pivot);
                double score = mapperScore(currentMapper);

                if (Math.abs(sortedFeature2[pivot] - sortedFeature2[pivot - 1]) < 1e-9) {
                    lastRes = new PartitionResult(pivot, score);
                    continue;
                }

                if (Math.abs(score - bestRes.getScore()) < 0.1) {
                    bestResults.add(new PartitionResult(pivot, score));
                }

                if (score > bestRes.getScore()) {
                    if (Math.abs(score - bestRes.getScore()) > 0.1) {
                        bestResults = new ArrayList<>();
                        bestResults.add(new PartitionResult(pivot, score));
                    }
                    bestRes.setScore(score);
                    bestRes.setSplitPosition(pivot);
                }

                lastRes = new PartitionResult(pivot, score);
            }
        }

        if (bestResults.size() == 1) {
            return bestRes;
        } else {
            //final double scoreRight = Math.log(end - split) + Math.log(split - start);
            bordersPtr = 0;
            double bs = -Double.MAX_VALUE;
            PartitionResult res = bestResults.get(0);
            for (int i = 0; i < bestResults.size(); i++) {
                PartitionResult r = bestResults.get(i);
                { // check that border doesn't exist
                    while (bordersPtr < bordersFeature2.size() && bordersFeature2.get(bordersPtr) < r.getSplitPosition()) {
                        bordersPtr++;
                    }
                }
                int end = sortedFeature2.length;
                int start = 1;
                if (bordersPtr < bordersFeature2.size())
                    end = bordersFeature2.get(bordersPtr);

                if (bordersPtr > 0)
                    start = bordersFeature2.get(bordersPtr - 1);

                final double scoreRight = Math.log(end - r.getSplitPosition()) + Math.log(r.getSplitPosition() - start);

                if (scoreRight > bs) {
                    bs = scoreRight;
                    res = r;
                }
            }
            return res;
        }
    }

    public static PartitionResultBigInt bestPartitionWithMapperBigInt(final int[] binNumberMapper,
                                                                      final double[] sortedFeature2,
                                                                      final TIntArrayList bordersFeature2) {
        int startPivot = firstPartition(bordersFeature2);
        //System.out.println("First pivot: " + startPivot);
        int bordersPtr = 0;
        HashMap<Integer, HashMap<Integer, Integer>> currentMapper = partitionCountersMapper(binNumberMapper, sortedFeature2, bordersFeature2, startPivot);
        PartitionResultBigInt bestRes = new PartitionResultBigInt(startPivot, mapperScoreBigInt(currentMapper));
        PartitionResultBigInt lastRes = new PartitionResultBigInt(startPivot, mapperScoreBigInt(currentMapper));

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

                BigInteger score = mapperScoreBigInt(currentMapper);

                if (Math.abs(sortedFeature2[pivot] - sortedFeature2[pivot - 1]) < 1e-9) {
                    lastRes = new PartitionResultBigInt(pivot, score);
                    continue;
                }

                if (score.compareTo(bestRes.getScore()) < 0) {
                    bestRes.setScore(score);
                    bestRes.setSplitPosition(pivot);
                }

                lastRes = new PartitionResultBigInt(pivot, score);
            } else {
                currentMapper = partitionCountersMapper(binNumberMapper, sortedFeature2, bordersFeature2, pivot);
                BigInteger score = mapperScoreBigInt(currentMapper);

                if (Math.abs(sortedFeature2[pivot] - sortedFeature2[pivot - 1]) < 1e-9) {
                    lastRes = new PartitionResultBigInt(pivot, score);
                    continue;
                }

                if (score.compareTo(bestRes.getScore()) < 0) {
                    bestRes.setScore(score);
                    bestRes.setSplitPosition(pivot);
                }

                lastRes = new PartitionResultBigInt(pivot, score);
            }
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
            if (score > bestScore) {
                bestScore = score;
                bestSplit = pivot;
            }
        }

        return new PartitionResult(bestSplit, bestScore);
    }
}
