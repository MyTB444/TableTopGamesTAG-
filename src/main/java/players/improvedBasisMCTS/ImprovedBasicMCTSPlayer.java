package players.improvedBasisMCTS;

import core.AbstractGameState;
import core.AbstractPlayer;
import core.actions.AbstractAction;
import core.interfaces.IStateHeuristic;

import java.util.List;
import java.util.Random;


/**
 * Simplified Optimized MCTS Player for 3-player Sushigo
 * Uses the existing ImprovedBasicTreeNode but with carefully tuned parameters
 *
 * Key optimizations for Sushigo:
 * - Tuned exploration constant for 3-player dynamics
 * - Short rollouts with strong heuristic reliance
 * - RAVE for faster learning in drafting games
 * - Progressive widening for variable hand sizes
 * - UCB1-Tuned for high-variance game
 */
public class ImprovedBasicMCTSPlayer extends AbstractPlayer {

    public ImprovedBasicMCTSPlayer() {
        this(System.currentTimeMillis());
    }

    public ImprovedBasicMCTSPlayer(long seed) {
        super(new ImprovedBasicMCTSParams(), "ImprovedSushigoMCTSPlayer");
        parameters.setRandomSeed(seed);
        rnd = new Random(seed);

        // Carefully tuned parameters for 3-player Sushigo

    }

    public ImprovedBasicMCTSPlayer(ImprovedBasicMCTSParams params) {
        super(params, "Simplified Sushigo MCTS");
        rnd = new Random(params.getRandomSeed());
    }

    /**
     * Configure parameters specifically optimized for 3-player Sushigo
     */


    @Override
    public AbstractAction _getAction(AbstractGameState gameState, List<AbstractAction> actions) {
        // Dynamically adjust parameters based on hand size (game phase)
        adjustParametersForGamePhase(actions.size());

        // Use the existing ImprovedBasicTreeNode with our tuned parameters
        ImprovedBasicTreeNode root = new ImprovedBasicTreeNode(this, null, gameState, rnd);

        // Perform MCTS search
        root.mctsSearch();

        // Return best action
        return root.bestAction();
    }

    /**
     * Dynamically adjust parameters based on game phase (hand size indicates phase)
     */
    private void adjustParametersForGamePhase(int handSize) {
        ImprovedBasicMCTSParams params = getParameters();

        if (handSize >= 8) {
            // Early in round: more exploration, longer rollouts
            params.K = 1.4;  // Higher exploration
            params.rolloutLength = 3;
            params.progressiveWideningC = 3.0;  // Explore more actions
            params.raveK = 500;  // Slower RAVE decay

        } else if (handSize >= 5) {
            // Mid-round: balanced approach
            params.K = 1.2;  // Balanced
            params.rolloutLength = 2;
            params.progressiveWideningC = 2.5;
            params.raveK = 400;

        } else if (handSize >= 3) {
            // Late round: more exploitation, focused search
            params.K = 1.0;  // More exploitation
            params.rolloutLength = 1;
            params.progressiveWideningC = 2.0;  // Focus on best actions
            params.raveK = 300;  // Faster RAVE decay

        } else {
            // Very late round: pure exploitation
            params.K = 0.7;  // Heavy exploitation
            params.rolloutLength = 0;  // Just use heuristic
            params.progressiveWideningC = 1.5;  // Very focused
            params.raveK = 200;
        }

        // Adjust tree depth based on hand size
        params.maxTreeDepth = Math.min(15, handSize * 2);
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
        return "SimplifiedSushigoMCTS";
    }

    @Override
    public ImprovedBasicMCTSPlayer copy() {
        return new ImprovedBasicMCTSPlayer((ImprovedBasicMCTSParams) getParameters().copy());
    }
}