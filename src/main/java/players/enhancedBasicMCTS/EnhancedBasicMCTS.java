package players.enhancedBasicMCTS;

import core.AbstractGameState;
import core.AbstractPlayer;
import core.actions.AbstractAction;
import core.interfaces.IStateHeuristic;

import java.util.List;
import java.util.Random;


/**
 * Improved Basic MCTS with minimal changes but higher win rate
 * Changes from original:
 * 1. Better default parameters
 * 2. Smarter action selection in rollouts
 * 3. Early termination on dominant actions
 * 4. Improved visit-based selection
 */
public class EnhancedBasicMCTS extends AbstractPlayer {

    public EnhancedBasicMCTS() {
        this(System.currentTimeMillis());
    }

    public EnhancedBasicMCTS(long seed) {
        super(new BasicMCTSParams(), "Improved Basic MCTS");
        parameters.setRandomSeed(seed);
        rnd = new Random(seed);

        // IMPROVEMENT 1: Better default parameters for Sushi Go
        BasicMCTSParams params = getParameters();
        params.K = 1.6;  // Lower K for better heuristic exploitation
        params.rolloutLength = 8;  // Shorter rollouts
        params.maxTreeDepth = 15;  // Deeper tree
        params.epsilon = 1e-6;
    }

    public EnhancedBasicMCTS(BasicMCTSParams params) {
        super(params, "Improved Basic MCTS");
        rnd = new Random(params.getRandomSeed());
    }

    @Override
    public AbstractAction _getAction(AbstractGameState gameState, List<AbstractAction> actions) {
        // Search for best action from the root
        ImprovedBasicTreeNode root = new ImprovedBasicTreeNode(this, null, gameState, rnd);

        // mctsSearch does all of the hard work
        root.mctsSearch();

        // Return best action
        return root.bestAction();
    }

    @Override
    public BasicMCTSParams getParameters() {
        return (BasicMCTSParams) parameters;
    }

    public void setStateHeuristic(IStateHeuristic heuristic) {
        getParameters().heuristic = heuristic;
    }

    @Override
    public String toString() {
        return "ImprovedBasicMCTS";
    }

    @Override
    public EnhancedBasicMCTS copy() {
        return new EnhancedBasicMCTS((BasicMCTSParams) parameters.copy());
    }
}