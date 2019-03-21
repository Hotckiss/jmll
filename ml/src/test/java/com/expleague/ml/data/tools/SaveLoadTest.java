package com.expleague.ml.data.tools;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.Pair;
import com.expleague.ml.func.Linear;
import com.expleague.ml.loss.L2;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.TargetMeta;
import com.expleague.ml.meta.impl.JsonDataSetMeta;
import com.expleague.ml.meta.items.FakeItem;
import com.expleague.ml.methods.LASSOMethod;
import java.util.HashMap;
import java.util.Map;
import org.junit.Assert;
import org.junit.Test;

import java.io.StringReader;
import java.io.StringWriter;
import java.util.Arrays;
import java.util.Date;
import java.util.function.Function;

public class SaveLoadTest {
  @Test
  public void testModel() {
    Pool.Builder<FakeItem> builder = Pool.builder(
        new JsonDataSetMeta("test", "me", new Date(), FakeItem.class, "unique-" + new FastRandom(100500).nextBase64String(10)),
        new MegaFeatureSet()
    );
    builder.accept(new FakeItem(1)); builder.advance();
    builder.accept(new FakeItem(2)); builder.advance();
    builder.accept(new FakeItem(3)); builder.advance();
    builder.accept(new FakeItem(4)); builder.advance();
    Pool<FakeItem> pool = builder.create();
    LASSOMethod method = new LASSOMethod(2, 1);
    Linear model = method.fit(pool.vecData(), pool.target(L2.class));
    StringWriter writer = new StringWriter();
    DataTools.writeModel(model, pool.features(), writer);
    Pair<Function, FeatureMeta[]> pair = DataTools.readModel(new StringReader(writer.toString()));
    Assert.assertTrue(Arrays.deepEquals(pair.second, pool.features()));
  }

  @Test
  public void appendPoolTest() {
    Pool.Builder<FakeItem> builder = Pool.builder(
        new JsonDataSetMeta("test", "me", new Date(), FakeItem.class, "unique-" + new FastRandom(100500).nextBase64String(10)),
        new MegaFeatureSet()
    );

    FakeItem[] trueItems = {new FakeItem(0), new FakeItem(1), new FakeItem(2), new FakeItem(3)};
    FeatureMeta[] megaMeta = {new MegaFeatureSet().meta(0), new MegaFeatureSet().meta(1)};

    builder.accept(trueItems[0]); builder.advance();
    builder.accept(trueItems[1]); builder.advance();
    builder.accept(trueItems[2], new ArrayVec(3, 42), megaMeta);
    builder.accept(trueItems[3], new ArrayVec(4, 42), megaMeta);
    Pool<FakeItem> pool = builder.create();

    Assert.assertTrue(Arrays.deepEquals(pool.items.toArray(), trueItems));
  }

  public static class MegaFeatureSet extends FeatureSet.Stub<FakeItem>{
    public static final FeatureMeta GREAT_FEATURE = FeatureMeta.create("great!", "Mega feature", FeatureMeta.ValueType.VEC);
    public static final TargetMeta SUPER_TARGET = TargetMeta.create("super!", "Super target", FeatureMeta.ValueType.VEC);

    @Override
    public Vec advanceTo(Vec to) {
      set(GREAT_FEATURE, 1);
      set(SUPER_TARGET, 42);
      return super.advanceTo(to);
    }
  }
}
