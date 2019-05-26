package com.expleague.ml.benchmark.generators;

import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.data.tools.FakePool;
import com.expleague.ml.data.tools.Pool;

public class FakePoolsGenerator {
    public static Pool<?> sameFeaturesPool(int xdim, int len, double duplicateProb) {
        double[] data = new double[xdim * len];
        int[] target = new int[len];
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < xdim; j++) {
                data[xdim * i + j] = i;
            }
            target[i] = i;
        }

        ArrayVec vec = new ArrayVec(data, 0, xdim * len);
        final VecBasedMx vdata = new VecBasedMx(xdim, vec);

        return FakePool.create(
                vdata,
                new IntSeq(target, 0, len)
        );
    }

    public static Pool<?> logFeaturesPool(int xdim1, int xdim2, int len, double duplicateProb) {
        int xdim = xdim1 + xdim2;
        double[] data = new double[xdim * len];
        int[] target = new int[len];
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < xdim1; j++) {
                data[xdim * i + j] = i;
            }

            for (int j = xdim1; j < xdim; j++) {
                data[xdim * i + j] = Math.log(i + 1);
            }

            target[i] = i;
        }

        ArrayVec vec = new ArrayVec(data, 0, xdim * len);
        final VecBasedMx vdata = new VecBasedMx(xdim, vec);

        return FakePool.create(
                vdata,
                new IntSeq(target, 0, len)
        );
    }
}
