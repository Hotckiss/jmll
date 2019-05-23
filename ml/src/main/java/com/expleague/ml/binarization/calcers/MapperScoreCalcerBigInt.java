package com.expleague.ml.binarization.calcers;

import java.math.BigInteger;
import java.util.HashMap;
import java.util.Map;

public class MapperScoreCalcerBigInt {
    public static BigInteger mapperScoreBigInt(HashMap<Integer, HashMap<Integer, Integer>> mapper) {
        BigInteger score = new BigInteger("1", 10);
        for (Map.Entry<Integer, HashMap<Integer, Integer>> entry : mapper.entrySet()) {
            for (Map.Entry<Integer, Integer> entry1 : entry.getValue().entrySet()) {
                for (int k =0; k < entry1.getValue(); k++) {
                    score = score.multiply(new BigInteger(String.valueOf((entry1.getValue() + 1)), 10));
                }
            }
        }

        return score;
    }
}
