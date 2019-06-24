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

    /**
     * Insert new border to array
     * @param borders
     * @param newBorder
     * @return
     */
    public static TIntArrayList insertBorder(final TIntArrayList borders, final int newBorder) {
        final TIntArrayList mergedBorders = new TIntArrayList();
        int i = 0;

        while (i < borders.size() && borders.get(i) < newBorder) {
            mergedBorders.add(borders.get(i));
            i++;
        }

        mergedBorders.add(newBorder);

        while (i < borders.size() && borders.get(i) == newBorder) {
            i++;
        }

        while (i < borders.size()) {
            mergedBorders.add(borders.get(i));
            i++;
        }

        return mergedBorders;
    }
}
