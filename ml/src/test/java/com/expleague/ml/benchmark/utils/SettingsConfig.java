package com.expleague.ml.benchmark.utils;

public class SettingsConfig {
    private int seed;
    private double learnSize;
    private int binFactor;
    private int treeDepth;
    private int iters;
    private double step;

    public SettingsConfig(final int seed,
                          final double learnSize,
                          final int binFactor,
                          final int treeDepth,
                          final int iters,
                          final double step) {
        this.seed = seed;
        this.learnSize = learnSize;
        this.binFactor = binFactor;
        this.treeDepth = treeDepth;
        this.iters = iters;
        this.step = step;
    }

    public SettingsConfig() {
        this.seed = 1;
        this.learnSize = 0.2;
        this.binFactor = 32;
        this.treeDepth = 6;
        this.iters = 2000;
        this.step = 0.005;
    }


    public int getSeed() {
        return seed;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }

    public double getLearnSize() {
        return learnSize;
    }

    public void setLearnSize(double learnSize) {
        this.learnSize = learnSize;
    }

    public int getBinFactor() {
        return binFactor;
    }

    public void setBinFactor(int binFactor) {
        this.binFactor = binFactor;
    }

    public int getTreeDepth() {
        return treeDepth;
    }

    public void setTreeDepth(int treeDepth) {
        this.treeDepth = treeDepth;
    }

    public int getIters() {
        return iters;
    }

    public void setIters(int iters) {
        this.iters = iters;
    }

    public double getStep() {
        return step;
    }

    public void setStep(double step) {
        this.step = step;
    }
}
