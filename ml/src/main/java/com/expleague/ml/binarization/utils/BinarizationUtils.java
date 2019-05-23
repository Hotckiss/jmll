package com.expleague.ml.binarization.utils;

import gnu.trove.list.array.TIntArrayList;

public class BinarizationUtils {
    /**
     * Finds first unused split position
     * @param bordersFeature2
     * @return
     */
    public static int firstPartition(final TIntArrayList bordersFeature2) {
        int res = 1;
        int bordersPtr = 0;
        while (bordersPtr < bordersFeature2.size() && bordersFeature2.get(bordersPtr) == res) {
            bordersPtr++;
            res++;
        }

        return res;
    }
}
