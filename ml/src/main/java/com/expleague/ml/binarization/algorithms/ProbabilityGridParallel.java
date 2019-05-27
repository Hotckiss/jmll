package com.expleague.ml.binarization.algorithms;

import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.ml.BFGrid;
import com.expleague.ml.BuildProgressHandler;
import com.expleague.ml.binarization.partitions.PartitionResult;
import com.expleague.ml.binarization.wrappers.PermutationWrapper;
import com.expleague.ml.binarization.wrappers.SortedFeatureWrapper;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.impl.BFGridImpl;
import com.expleague.ml.impl.BFRowImpl;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;

import static com.expleague.ml.binarization.partitions.BestPartitionsSearchers.bestPartitionWithMapper_veryFast;
import static com.expleague.ml.binarization.utils.BinarizationUtils.insertBorder;
import static com.expleague.ml.binarization.utils.MappersUtils.buildBinsMapper;

public class ProbabilityGridParallel {
    /**
     * Builds naive probability grid
     * @param ds
     * @param binFactor
     * @return
     */
    public static BFGrid probabilityGrid(final VecDataSet ds, final int binFactor, BuildProgressHandler buildProgressHandler) {
        assert (binFactor < ds.length());

        final int dim = ds.xdim();
        final BFRowImpl[] rows = new BFRowImpl[dim];
        int bfCount = 0;
        ArrayList<TIntArrayList> currentBorders = new ArrayList<>();

        ArrayList<SortedFeatureWrapper> sortedFeatures = new ArrayList<>();
        ArrayList<PermutationWrapper> orders = new ArrayList<>();
        ArrayList<PermutationWrapper> reverces = new ArrayList<>();

        for (int feature_index = 0; feature_index < dim; feature_index++) {
            final double[] feature = new double[ds.length()];
            final ArrayPermutation permutation = new ArrayPermutation(ds.order(feature_index));
            final int[] order = permutation.direct();
            final int[] reverse = permutation.reverse();
            for (int i = 0; i < feature.length; i++)
                feature[i] = ds.at(order[i]).get(feature_index);

            sortedFeatures.add(new SortedFeatureWrapper(feature));
            orders.add(new PermutationWrapper(order));
            reverces.add(new PermutationWrapper(reverse));
        }

        //initial borders
        for (int i = 0; i < dim; i++) {
            TIntArrayList borders = new TIntArrayList();
            borders.add(ds.length());
            currentBorders.add(borders);
        }

        for(int iters = 0; iters < binFactor; iters++) {
            for (int feature_index = 0; feature_index < dim; feature_index++) {
                final double[] feature = sortedFeatures.get(feature_index).sortedFeature;
                final int[] order = orders.get(feature_index).permutation;
                final int[] reverse = reverces.get(feature_index).permutation;
                boolean haveDiffrentElements = false;
                for (int i = 1; i < order.length; i++)
                    if (order[i] != order[0])
                        haveDiffrentElements = true;
                if (!haveDiffrentElements)
                    continue;

                PartitionResult bestFromAll = PartitionResult.makeWorst();

                for (int paired_feature_index = 0; paired_feature_index < dim; paired_feature_index += 4) {
                    buildProgressHandler.step();
                    buildProgressHandler.step();
                    buildProgressHandler.step();
                    buildProgressHandler.step();
                    Thread t1 = null;
                    Thread t2 = null;
                    Thread t3 = null;
                    Thread t4 = null;
                    final int fi = feature_index;

                    if (paired_feature_index != feature_index) {
                        final int p = paired_feature_index;
                        t1 = new Thread(() -> {
                            final int[] reversePaired = reverces.get(p).permutation;

                            int[] binNumberMapper = buildBinsMapper(currentBorders.get(p), reverse, reversePaired);

                            PartitionResult bestResult = PartitionResult.makeWorst();

                            bestResult = bestPartitionWithMapper_veryFast(binNumberMapper, feature, currentBorders.get(fi));

                            synchronized (bestFromAll) {
                                if (bestFromAll.getScore() < bestResult.getScore()) {
                                    bestFromAll.setScore(bestResult.getScore());
                                    bestFromAll.setSplitPosition(bestResult.getSplitPosition());
                                }
                            }
                        });

                    }

                    if (paired_feature_index + 1 < dim && paired_feature_index + 1 != feature_index) {
                        final int p = paired_feature_index + 1;
                        t2 = new Thread(() -> {
                            final int[] reversePaired = reverces.get(p).permutation;

                            int[] binNumberMapper = buildBinsMapper(currentBorders.get(p), reverse, reversePaired);

                            PartitionResult bestResult = PartitionResult.makeWorst();

                            bestResult = bestPartitionWithMapper_veryFast(binNumberMapper, feature, currentBorders.get(fi));

                            synchronized (bestFromAll) {
                                if (bestFromAll.getScore() < bestResult.getScore()) {
                                    bestFromAll.setScore(bestResult.getScore());
                                    bestFromAll.setSplitPosition(bestResult.getSplitPosition());
                                }
                            }
                        });
                    }

                    if (paired_feature_index + 2 < dim && paired_feature_index + 2 != feature_index) {
                        final int p = paired_feature_index + 2;
                        t3 = new Thread(() -> {
                            final int[] reversePaired = reverces.get(p).permutation;

                            int[] binNumberMapper = buildBinsMapper(currentBorders.get(p), reverse, reversePaired);

                            PartitionResult bestResult = PartitionResult.makeWorst();

                            bestResult = bestPartitionWithMapper_veryFast(binNumberMapper, feature, currentBorders.get(fi));

                            synchronized (bestFromAll) {
                                if (bestFromAll.getScore() < bestResult.getScore()) {
                                    bestFromAll.setScore(bestResult.getScore());
                                    bestFromAll.setSplitPosition(bestResult.getSplitPosition());
                                }
                            }
                        });
                    }

                    if (paired_feature_index + 3 < dim && paired_feature_index + 3 != feature_index) {
                        final int p = paired_feature_index + 3;
                        t4 = new Thread(() -> {
                            final int[] reversePaired = reverces.get(p).permutation;

                            int[] binNumberMapper = buildBinsMapper(currentBorders.get(p), reverse, reversePaired);

                            PartitionResult bestResult = PartitionResult.makeWorst();

                            bestResult = bestPartitionWithMapper_veryFast(binNumberMapper, feature, currentBorders.get(fi));

                            synchronized (bestFromAll) {
                                if (bestFromAll.getScore() < bestResult.getScore()) {
                                    bestFromAll.setScore(bestResult.getScore());
                                    bestFromAll.setSplitPosition(bestResult.getSplitPosition());
                                }
                            }
                        });
                    }

                    if (t1 != null) {
                        t1.start();
                    }

                    if (t2 != null) {
                        t2.start();
                    }

                    if (t3 != null) {
                        t3.start();
                    }

                    if (t4 != null) {
                        t4.start();
                    }

                    try {
                        if (t1 != null) {
                            t1.join();
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    try {
                        if (t2 != null) {
                            t2.join();
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    try {
                        if (t3 != null) {
                            t3.join();
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    try {
                        if (t4 != null) {
                            t4.join();
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }

                if (bestFromAll.getSplitPosition() > 1) {
                    TIntArrayList newBorders = insertBorder(currentBorders.get(feature_index), bestFromAll.getSplitPosition());
                    currentBorders.set(feature_index, newBorders);
                }

            }
        }

        System.out.print("[");
        for (int i = 0; i < currentBorders.size(); i++) {
            System.out.print("[");
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < currentBorders.get(i).size() - 1; j++) {
                sb.append(currentBorders.get(i).get(j) + ", ");
            }
            if (sb.length() > 0) {
                sb.delete(sb.length() - 2, sb.length());
            }
            System.out.println(sb.toString() + "], ");
        }
        System.out.println("]");
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
}
