package com.expleague.ml.binarization.algorithms;

import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.ml.BFGrid;
import com.expleague.ml.BuildProgressHandler;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.impl.BFGridImpl;
import com.expleague.ml.impl.BFRowImpl;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

public class EqualWidthBinarization {
    public static BFGrid equalWidthGrid(final VecDataSet ds, final int binFactor, BuildProgressHandler buildProgressHandler) {
        final int dim = ds.xdim();
        final BFRowImpl[] rows = new BFRowImpl[dim];
        int bfCount = 0;

        final double[] feature = new double[ds.length()];
        for (int f = 0; f < dim; f++) {
            buildProgressHandler.step();
            final ArrayPermutation permutation = new ArrayPermutation(ds.order(f));
            final int[] order = permutation.direct();

            for (int i = 0; i < feature.length; i++) {
                feature[i] = ds.at(order[i]).get(f);
            }

            double min = feature[0];
            double max = feature[feature.length - 1];
            double width = (max - min) / binFactor;

            final TIntArrayList borders = new TIntArrayList();

            for (int i = 1; i < feature.length; i++) {
                if (binNumber(min, feature[i], width) != binNumber(min, feature[i - 1], width)) {
                    borders.add(i);
                }
            }

            borders.add(ds.length());
            final TDoubleArrayList dborders = new TDoubleArrayList();
            final TIntArrayList sizes = new TIntArrayList();
            int size = borders.size();
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

    private static int binNumber(double min, double val, double width) {
        return  (int)Math.round((val - min) / width);
    }
}
