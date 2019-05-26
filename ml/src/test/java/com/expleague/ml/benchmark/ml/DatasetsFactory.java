package com.expleague.ml.benchmark.ml;

import com.expleague.ml.benchmark.generators.FakePoolsGenerator;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.testUtils.TestResourceLoader;

import java.io.IOException;

public class DatasetsFactory {
    public static Pool<?> makePool(DatasetType type) throws IOException {
        switch (type) {
            case FEATURES_TXT:
                return TestResourceLoader.loadPool("features.txt");
            case SAME:
                return FakePoolsGenerator.sameFeaturesPool(50, 12000);
            case SAME_LOG:
                return FakePoolsGenerator.logFeaturesPool(25, 25, 12000);
            case SAME_DUPL_0_25:
                return FakePoolsGenerator.sameFeaturesPoolDupl(50, 12000, 0.25);
            case RANDOM_FUNCS:
                return FakePoolsGenerator.randomFuncsPool(50 , 12000);
        }

        return TestResourceLoader.loadPool("features.txt");
    }

    public static String getDatasetName(DatasetType type) {
        switch (type) {
            case FEATURES_TXT:
                return "Features.txt";
            case SAME:
                return "Same features";
            case SAME_LOG:
                return "Features + log";
            case SAME_DUPL_0_25:
                return "Same features + duplicates";
            case RANDOM_FUNCS:
                return "Random functions";
        }

        return "Median division";
    }

    public static DatasetType getDatasetType(String raw) {
        switch (raw) {
            case "Features.txt":
                return DatasetType.FEATURES_TXT;
            case "Same features":
                return DatasetType.SAME;
            case "Features + log":
                return DatasetType.SAME_LOG;
            case "Same features + duplicates":
                return DatasetType.SAME_DUPL_0_25;
            case "Random functions":
                return DatasetType.RANDOM_FUNCS;
            default:
                return DatasetType.FEATURES_TXT;
        }
    }

    public static String[] getDatasetNames() {
        String[] res = new String[DatasetType.values().length];
        int ptr = 0;
        for (DatasetType type : DatasetType.values()) {
            res[ptr++] = getDatasetName(type);
        }

        return res;
    }
}
