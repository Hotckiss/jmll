package com.expleague.ml.binarization.algorithms;

import com.expleague.ml.BFGrid;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.impl.BFGridImpl;
import com.expleague.ml.impl.BFRowImpl;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class EntropyDiscretization {
    private Pool<?> learn;
    private int binFactor;
    private int[] classes;
    private TIntArrayList[] borders;

    public EntropyDiscretization(Pool<?> learn, int binFactor) {
        this.learn = learn;
        this.binFactor = binFactor;

        int targetsLength = learn.target(0).length();
        double[] targets = new double[targetsLength];
        double[] sortedTargets = new double[targetsLength];
        for (int i = 0; i < targetsLength; i++) {
            targets[i] = Double.valueOf(learn.target(0).at(i).toString());
            sortedTargets[i] = targets[i];
        }

        Arrays.sort(sortedTargets);

        double min = sortedTargets[0];
        double max = sortedTargets[targetsLength - 1];

        double width = (max - min) / binFactor;

        int[] classes = new int[targetsLength];

        for (int i = 0; i < targetsLength; i++) {
            classes[i] = binNumber(min, targets[i], width);
        }

        this.classes = classes;
        this.borders = new TIntArrayList[learn.vecData().xdim()];
        for (int i = 0; i < learn.vecData().xdim(); i++) {
            borders[i] = new TIntArrayList();
        }

        /*this.classes = new int[]{0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1};
        binarizeFeature(new double[]{53, 56, 57, 63, 66, 67, 67, 67, 68, 69, 70, 70, 70, 70, 72, 73, 75 ,75, 76, 76, 78, 79, 80, 81}, 0);
        System.out.println(borders[0].size());
        for(int i = 0; i < borders[0].size(); i++) {
            System.out.println(borders[0].get(i));
        }*/
    }

    public BFGrid fit() {
        int f = 0;
        double[] rawFeature = new double[learn.vecData().length()];
        for (int j = 0; j < learn.vecData().length(); j++) {
            rawFeature[j] = learn.vecData().data().get(j, f);
        }

        Arrays.sort(rawFeature);
        binarizeFeature(rawFeature, f);
        borders[0].add(learn.vecData().length());

        for(int fe = 1; fe < learn.vecData().xdim(); fe++) {
            for (int i = 0; i < borders[0].size(); i++) {
                borders[fe].add(borders[0].get(i));
            }
        }

        int bfCount = 0;
        final BFRowImpl[] rows = new BFRowImpl[learn.vecData().xdim()];
        for(int fe = 0; fe < learn.vecData().xdim(); fe++) {
            final TDoubleArrayList dborders = new TDoubleArrayList();
            final TIntArrayList sizes = new TIntArrayList();
            { // drop existing
                int size = borders[fe].size();
                for (int b = 0; b < size - 1; b++) {
                    int borderValue = borders[fe].get(b);
                    double[] feature = new double[learn.vecData().length()];
                    for (int j = 0; j < learn.vecData().length(); j++) {
                        feature[j] = learn.vecData().data().get(j, fe);
                    }

                    Arrays.sort(rawFeature);
                    dborders.add((feature[borderValue - 1] + feature[borderValue]) / 2.);
                    sizes.add(borderValue);
                }
            }

            rows[fe] = new BFRowImpl(bfCount, fe, dborders.toArray(), sizes.toArray());

            bfCount += dborders.size();
        }

        return new BFGridImpl(rows);
    }

    private void binarizeFeature(double[] rawFeature, int f) {
        binarizeFeatureRec(rawFeature, 0, rawFeature.length, f);
    }

    private void binarizeFeatureRec(double[] rawFeature, int l, int r, int f) {
        double bestScore = Double.MAX_VALUE;
        int bestSplit = -1;

        for (int p = l; p < r; p++) {
            if (borders[f].contains(p)) {
                continue;
            }

            double score = ((double) (p - l) / (r - l)) * subsetEntropy(rawFeature, l, p) + ((double) (r - p) / (r - l)) * subsetEntropy(rawFeature, p, r);

            if (score < bestScore) {
                bestScore = score;
                bestSplit = p;
            }
        }

        int N = r - l;

        //System.out.println("------");
        //System.out.println(bestSplit);
        //System.out.println(gain(rawFeature, l, r, bestScore));
        //System.out.println((Math.log(N - 1) / Math.log(2)) / N + delta(rawFeature, l, r, bestSplit) / N);
        //System.out.println("------");
        if (gain(rawFeature, l, r, bestScore) > ((Math.log(N - 1) / Math.log(2)) / N) + (delta(rawFeature, l, r, bestSplit) / N) && bestSplit != -1) {
            borders[f].add(bestSplit);
            binarizeFeatureRec(rawFeature, l, bestSplit, f);
            binarizeFeatureRec(rawFeature, bestSplit, r, f);
        } else {
            return;
        }
    }

    private double gain(final double[] feature, int l, int r, double score) {
        return subsetEntropy(feature, l, r) - score;
    }

    private double delta(final double[] feature, int l, int r, int p) {
        HashMap<Integer, Boolean> represented = new HashMap<>();
        HashMap<Integer, Boolean> represented_l = new HashMap<>();
        HashMap<Integer, Boolean> represented_r = new HashMap<>();
        for (int i = l; i < p; i++) {
            represented.put(classes[i], true);
            represented_l.put(classes[i], true);
        }

        for (int i = p; i < r; i++) {
            represented.put(classes[i], true);
            represented_r.put(classes[i], true);
        }

        int k = represented.size();
        int k1 = represented_l.size();
        int k2 = represented_r.size();

        return Math.log(Math.pow(3, k) - 2) / Math.log(2) - (k*subsetEntropy(feature, l, r) - k1 * subsetEntropy(feature, l, p) - k2 * subsetEntropy(feature, p, r));
    }

    // [l; r)
    private double subsetEntropy(final double[] feature, int l, int r) {
        HashMap<Integer, Integer> counters = new HashMap<>();
        int size = r - l;
        double sum = 0;
        for (int i = l; i < r; i++) {
            int cls = classes[i];
            counters.merge(cls, 1, (a, b) -> a + b);
        }

        for (Map.Entry<Integer, Integer> entry : counters.entrySet()) {
            double pr = (double)entry.getValue() / size;
            sum += pr * (Math.log(pr) / Math.log(2));
        }

        return -sum;
    }

    private static int binNumber(double min, double val, double width) {
        return  (int)Math.round((val - min) / width);
    }
}
