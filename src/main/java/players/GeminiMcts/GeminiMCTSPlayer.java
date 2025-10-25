package players.GeminiMcts;

import core.AbstractGameState;
import core.AbstractPlayer;
import core.actions.AbstractAction;
import core.interfaces.IStateHeuristic;

import java.util.List;
import java.util.Random;

public class GeminiMCTSPlayer extends AbstractPlayer {

    public GeminiMCTSPlayer() {
        this(System.currentTimeMillis());
    }

    public GeminiMCTSPlayer(long seed) {
        super(new GeminiMCTSParams(), "Gemini MCTS");
        parameters.setRandomSeed(seed);
        rnd = new Random(seed);
    }

    public GeminiMCTSPlayer(GeminiMCTSParams params) {
        super(params, "Gemini MCTS");
        rnd = new Random(params.getRandomSeed());
    }

    @Override
    public AbstractAction _getAction(AbstractGameState gameState, List<AbstractAction> actions) {
        // Search for best action from the root
        GeminiMCTSTreeNode root = new GeminiMCTSTreeNode(this, null, gameState, rnd);

        // mctsSearch does all of the hard work
        root.mctsSearch();

        // Return best action
        return root.bestAction();
    }

    @Override
    public GeminiMCTSParams getParameters() {
        return (GeminiMCTSParams) parameters;
    }

    public void setStateHeuristic(IStateHeuristic heuristic) {
        getParameters().heuristic = heuristic;
    }

    @Override
    public String toString() {
        return "GeminiMCTS[K=" + getParameters().K + ",r=" + getParameters().rolloutLength + "]";
    }

    @Override
    public GeminiMCTSPlayer copy() {
        return new GeminiMCTSPlayer((GeminiMCTSParams) parameters.copy());
    }
}
