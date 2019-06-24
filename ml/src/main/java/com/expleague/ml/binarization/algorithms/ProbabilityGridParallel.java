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
    public static BFGrid probabilityGrid(final VecDataSet ds, final int binFactor, BuildProgressHandler buildProgressHandler, final int threadsCount) {
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

                for (int paired_feature_index = 0; paired_feature_index < dim; paired_feature_index += threadsCount) {
                    ArrayList<Thread> threads = new ArrayList<>();
                    final int fi = feature_index;
                    for (int t = 0; t < threadsCount; t++) {
                        final int pi = paired_feature_index + t;
                        if (pi >= dim) continue;
                        buildProgressHandler.step();
                        if (pi == fi) continue;
                        threads.add(new Thread(() -> {
                            final int[] reversePaired = reverces.get(pi).permutation;

                            int[] binNumberMapper = buildBinsMapper(currentBorders.get(pi), reverse, reversePaired);

                            PartitionResult bestResult = PartitionResult.makeWorst();

                            bestResult = bestPartitionWithMapper_veryFast(binNumberMapper, feature, currentBorders.get(fi));

                            synchronized (bestFromAll) {
                                if (bestFromAll.getScore() < bestResult.getScore()) {
                                    bestFromAll.setScore(bestResult.getScore());
                                    bestFromAll.setSplitPosition(bestResult.getSplitPosition());
                                }
                            }
                        }));
                    }

                    for (Thread th : threads) {
                        th.start();
                    }

                    for (Thread th : threads) {
                        try {
                            th.join();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
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
