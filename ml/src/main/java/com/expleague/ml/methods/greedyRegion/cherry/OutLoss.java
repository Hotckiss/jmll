package com.expleague.ml.methods.greedyRegion.cherry;

import com.expleague.ml.data.cherry.CherryLoss;
import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.ml.data.cherry.CherryPointsHolder;
import com.expleague.ml.BFGrid;
import com.expleague.ml.impl.BFRowImpl;
import com.expleague.ml.loss.StatBasedLoss;
import gnu.trove.set.hash.TIntHashSet;

import static com.expleague.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

public class OutLoss<Subset extends CherryPointsHolder, Loss extends StatBasedLoss<AdditiveStatistics>> extends CherryLoss {
  private Subset subset;
  private Loss loss;
  private int complexity = 1;
  private int minBinSize = 50;
  private TIntHashSet used = new TIntHashSet();

  OutLoss(Subset subset, Loss loss) {
    this.subset = subset;
    this.loss = loss;
  }

  @Override
  public double score(BFGrid.Row feature, int start, int end, AdditiveStatistics added, AdditiveStatistics out) {
    if (start == 0 && end == feature.size())
      return Double.NEGATIVE_INFINITY;
    int newsize = used.contains(feature.findex()) ? used.size() : used.size()+1;
    if (newsize > 7)
      return Double.NEGATIVE_INFINITY;
    AdditiveStatistics inside = subset.inside().append(added);
    final int borders = borders(feature, start, end);
    return score(inside, out, complexity + borders);
  }

  private int borders(BFGrid.Row feature, int start, int end) {
    return start != 0 && end != feature.size() ? 4 : 1;
  }

  private double score(AdditiveStatistics inside, AdditiveStatistics outside, int complexity) {
    final double wIn = weight(inside);
    if (used.size() > 6)
      return Double.NEGATIVE_INFINITY;
    if (wIn > 0 && wIn < minBinSize)
      return -1000000;
    final double wOut = weight(outside);
    if (wOut > 0 && wOut < minBinSize)
      return -1000000;
    return -loss.score(inside) / complexity;
  }

  @Override
  public double score() {
    return score(subset.inside(), subset.outside(), complexity);
  }

  @Override
  public double insideIncrement() {
    return loss.bestIncrement(subset.inside());
  }

  @Override
  public void endClause() {
    subset.endClause();
    complexity ++;
  }

  public void addCondition(BFRowImpl feature, int start, int end) {
    subset().addCondition(feature, start, end);
    complexity += borders(feature, start, end);
    used.add(feature.origFIndex);
    complexity ++;
  }

  @Override
  public CherryPointsHolder subset() {
    return subset;
  }
}

