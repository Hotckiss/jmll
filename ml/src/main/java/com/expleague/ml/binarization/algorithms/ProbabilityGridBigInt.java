package com.expleague.ml.binarization.algorithms;

import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.ml.BFGrid;
import com.expleague.ml.BuildProgressHandler;
import com.expleague.ml.binarization.partitions.PartitionResultBigInt;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.impl.BFGridImpl;
import com.expleague.ml.impl.BFRowImpl;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;

import static com.expleague.ml.binarization.partitions.BestPartitionsSearchers.bestPartitionWithMapperBigInt;
import static com.expleague.ml.binarization.utils.BinarizationUtils.insertBorder;
import static com.expleague.ml.binarization.utils.MappersUtils.buildBinsMapper;

/**
 * ProbabilityGrid with big int score criteria
 */
public class ProbabilityGridBigInt {
    public static BFGrid probabilityGrid_bigInt(final VecDataSet ds, final int binFactor, BuildProgressHandler buildProgressHandler) {
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

                PartitionResultBigInt bestFromAll = PartitionResultBigInt.makeWorst();

                for (int paired_feature_index = 0; paired_feature_index < dim; paired_feature_index++) {
                    buildProgressHandler.step();
                    if (paired_feature_index == feature_index) {
                        continue;
                    }

                    System.out.println(paired_feature_index);

                    final ArrayPermutation permutationPaired = new ArrayPermutation(ds.order(paired_feature_index));
                    final int[] reversePaired = permutationPaired.reverse();

                    int[] binNumberMapper = buildBinsMapper(currentBorders.get(paired_feature_index), reverse, reversePaired);

                    PartitionResultBigInt bestResult = bestPartitionWithMapperBigInt(binNumberMapper, feature, currentBorders.get(feature_index));

                    if (bestResult.getScore().compareTo(bestFromAll.getScore()) < 0) {
                        bestFromAll = bestResult;
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
