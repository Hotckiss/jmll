package com.expleague.ml.binarization.partitions;

public class PartitionResult {
    private int splitPosition;
    private double score;

    public PartitionResult(int splitPosition, double score) {
        this.score = score;
        this.splitPosition = splitPosition;
    }

    public int getSplitPosition() {
        return splitPosition;
    }

    public double getScore() {
        return score;
    }

    public void setSplitPosition(int newSplitPosition) {
        this.splitPosition = newSplitPosition;
    }

    public void setScore(double newScore) {
        this.score = newScore;
    }

    public static PartitionResult makeWorst() {
        return new PartitionResult(-1, -Double.MAX_VALUE);
    }
}
