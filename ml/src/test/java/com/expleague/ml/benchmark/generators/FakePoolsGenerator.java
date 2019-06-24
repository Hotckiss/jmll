package com.expleague.ml.benchmark.generators;

import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.data.tools.FakePool;
import com.expleague.ml.data.tools.Pool;

import java.util.Arrays;
import java.util.Random;

public class FakePoolsGenerator {
    public static Pool<?> sameFeaturesPool(int xdim, int len) {
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

    public static Pool<?> logFeaturesPool(int xdim1, int xdim2, int len) {
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

    public static Pool<?> sameFeaturesPoolDupl(int xdim, int len, double duplicateProb) {
        double[] data = new double[xdim * len];
        Random r = new Random();
        int[] target = new int[len];
        for (int i = 0; i < len; i++) {
            double rnd = r.nextDouble();
            if (i > 0 && rnd < duplicateProb) {
                for (int j = 0; j < xdim; j++) {
                    data[xdim * i + j] = data[xdim * (i - 1) + j];
                }
                target[i] = target[i - 1];
            } else {
                for (int j = 0; j < xdim; j++) {
                    data[xdim * i + j] = i;
                }
                target[i] = i;
            }

        }

        ArrayVec vec = new ArrayVec(data, 0, xdim * len);
        final VecBasedMx vdata = new VecBasedMx(xdim, vec);

        return FakePool.create(
                vdata,
                new IntSeq(target, 0, len)
        );
    }

    public static Pool<?> randomFuncsPool(int xdim, int len) {
        double[] data = new double[xdim * len];
        Random r = new Random();
        int[] target = new int[len];
        for (int i = 0; i < len; i++) {
            double[] cur = new double[xdim];
            double rnd = r.nextDouble() * 10;
            double val = i;
            if (rnd < 1) {
                val = i;
            } else if (rnd < 2) {
                val = i + 10;
            } else if (rnd < 3) {
                val = Math.log(i + 1);
            } else if (rnd < 4) {
                val = Math.log(i * i);
            } else if (rnd < 5) {
                val = i * i;
            } else if (rnd < 6) {
                val = 5 * i;
            } else if (rnd < 7) {
                val = Math.sqrt(i);
            } else if (rnd < 8) {
                val = Math.sin(i) * i;
            } else if (rnd < 9) {
                val = Math.atan(i);
            } else if (rnd < 10) {
                val = i * i - 10*i + 34;
            } else {
                val = i;
            }

            for (int j = 0; j < xdim; j++) {
                data[xdim * i + j] = val;
                cur[j] = val;
            }

            Arrays.sort(cur);
            target[i] = (int)cur[xdim / 2];
        }

        ArrayVec vec = new ArrayVec(data, 0, xdim * len);
        final VecBasedMx vdata = new VecBasedMx(xdim, vec);

        return FakePool.create(
                vdata,
                new IntSeq(target, 0, len)
        );
    }
}
