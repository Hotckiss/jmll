package com.expleague.ml;

import java.util.HashMap;
import java.util.Map;

public class GridUtils {
    public static HashMap<Integer, Integer> counts = new HashMap<>();
    public static int total = 0;

    public static void inc(int binNo) {
        Integer old = counts.get(binNo);

        if (old == null) {
            counts.put(binNo, 1);
        } else {
            counts.put(binNo, old + 1);
        }

        total++;
    }

    public static void out() {
        System.out.println("Total: " + total);

        for (Map.Entry<Integer, Integer> entry1 : counts.entrySet()) {
            System.out.println("Bin: " + entry1.getKey() + " Count: " + entry1.getValue());
        }
    }

    public static void outArr() {
        System.out.print("[");
        for (Map.Entry<Integer, Integer> entry1 : counts.entrySet()) {
            System.out.print(entry1.getValue() + ", ");
        }
        System.out.println("]");
    }
}
