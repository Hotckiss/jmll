package learning.transform;

import learning.LearningState;

public class ReplaceTransitionTransform<T> implements Transform<T> {
  private final int from;
  private final int to;
  private final T c;

  public ReplaceTransitionTransform(int from, int to, T c) {
    this.from = from;
    this.to = to;
    this.c = c;
  }

  @Override
  public LearningState<T> applyTransform(LearningState<T> learningState) {
    LearningState<T> state1 = new RemoveTransitionTransform<>(from, c).applyTransform(learningState);
    return new AddTransitionTransform<>(from, to, c).applyTransform(state1);
  }

  @Override
  public String getDescription() {
    return String.format("Redirect edge from %d by %s to %d", from, c.toString(), to);
  }
}
