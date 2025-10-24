package players.improvedBasisMCTS;

import core.AbstractGameState;
import core.AbstractPlayer;
import core.actions.AbstractAction;
import core.interfaces.IStateHeuristic;

import java.util.List;
import java.util.Random;


/**
 * Improved version of Basic MCTS with several enhancements:
 * - RAVE (Rapid Action Value Estimation) for faster learning
 * - Progressive widening to manage large action spaces
 * - UCB1-Tuned for better exploration/exploitation balance
 * - Action ordering heuristic for better expansion
 */
public class ImprovedBasicMCTSPlayer extends AbstractPlayer {

    public ImprovedBasicMCTSPlayer() {
        this(System.currentTimeMillis());
    }

    public ImprovedBasicMCTSPlayer(long seed) {
        super(new ImprovedBasicMCTSParams(), "Improved Basic MCTS");
        parameters.setRandomSeed(seed);
        rnd = new Random(seed);

        // Enhanced parameters for better performance
        ImprovedBasicMCTSParams params = getParameters();
        params.K = 0.7;  // Lower exploration for more focused search
        params.rolloutLength = 0;  // Trust the heuristic
        params.maxTreeDepth = 20;  // Deeper tree for better planning
        params.epsilon = 1e-6;

        // RAVE parameters
        params.useRAVE = true;
        params.raveK = 300;  // RAVE bias decreases with visits

        // Progressive widening
        params.useProgressiveWidening = true;
        params.progressiveWideningAlpha = 0.5;
        params.progressiveWideningC = 2.0;

        // UCB variant
        params.useUCB1Tuned = false;  // Standard UCB1 works better for most games
    }

    public ImprovedBasicMCTSPlayer(ImprovedBasicMCTSParams params) {
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
    public ImprovedBasicMCTSParams getParameters() {
        return (ImprovedBasicMCTSParams) parameters;
    }

    public void setStateHeuristic(IStateHeuristic heuristic) {
        getParameters().heuristic = heuristic;
    }

    @Override
    public String toString() {
        return "ImprovedBasicMCTSAssignment";
    }

    @Override
    public ImprovedBasicMCTSPlayer copy() {
        return new ImprovedBasicMCTSPlayer((ImprovedBasicMCTSParams) parameters.copy());
    }
}