package com.expleague.ml.binarization.algorithms;

import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.ml.BFGrid;
import com.expleague.ml.BuildProgressHandler;
import com.expleague.ml.GridTools;
import com.expleague.ml.binarization.partitions.PartitionResult;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.impl.BFGridImpl;
import com.expleague.ml.impl.BFRowImpl;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;

import static com.expleague.ml.binarization.partitions.BestPartitionsSearchers.bestPartition;
import static com.expleague.ml.binarization.partitions.BestPartitionsSearchers.bestPartitionWithMapper;
import static com.expleague.ml.binarization.utils.BinarizationUtils.insertBorder;
import static com.expleague.ml.binarization.utils.MappersUtils.buildBinsMapper;

/**
 * Probability grid with median warming
 */
public class ProbabilityGridMedianWarming {
    public static BFGrid probabilityGrid_mixed(final VecDataSet ds, final int binFactor, boolean useFastAlgorithm, BuildProgressHandler buildProgressHandler) {
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
            for (int feature_index = 0; feature_index < dim; feature_index++) {
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

                if (feature_index < 7) {
                    for (int k = 0; k < dim; k++) {
                        buildProgressHandler.step();
                    }
                    currentBorders.set(feature_index, GridTools.greedyLogSumBorders(feature, binFactor));
                    continue;
                }

                PartitionResult bestFromAll = PartitionResult.makeWorst();

                int bf = 0;
                for (int paired_feature_index = 0; paired_feature_index < dim; paired_feature_index++) {
                    buildProgressHandler.step();
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

                    if (bestFromAll.getScore() < bestResult.getScore()) {
                        bestFromAll = bestResult;
                        bf = paired_feature_index;
                    }
                }

                if (bestFromAll.getSplitPosition() > 1) {
                    TIntArrayList newBorders = insertBorder(currentBorders.get(feature_index), bestFromAll.getSplitPosition());
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
}
