package com.expleague.ml.binarization.partitions;

import java.math.BigInteger;

public class PartitionResultBigInt {
    private int splitPosition;
    private BigInteger score;

    public PartitionResultBigInt(int splitPosition, BigInteger score) {
        this.score = score;
        this.splitPosition = splitPosition;
    }

    public int getSplitPosition() {
        return splitPosition;
    }

    public BigInteger getScore() {
        return score;
    }

    public void setSplitPosition(int newSplitPosition) {
        this.splitPosition = newSplitPosition;
    }

    public void setScore(BigInteger newScore) {
        this.score = newScore;
    }

    public static PartitionResultBigInt makeWorst() {
        StringBuilder sb = new StringBuilder();
        sb.append(1);
        for (int i = 0; i < 1000; i++) {
            sb.append(0);
        }
        return new PartitionResultBigInt(-1, new BigInteger(sb.toString(), 10));
    }
}
