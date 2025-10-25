package players.GeminiMcts; // Updated package

import core.AbstractGameState;
import core.actions.AbstractAction;
import core.interfaces.IStateHeuristic;
import games.sushigo.SGGameState; // Make sure this import path is correct
import players.PlayerConstants;
import utilities.ElapsedCpuTimer;

import java.util.*;

import static players.PlayerConstants.*;
import static utilities.Utils.noise;

public class GeminiMCTSTreeNode {

    protected GeminiMCTSPlayer player;
    protected GeminiMCTSTreeNode parent;
    protected GeminiMCTSTreeNode root;
    protected AbstractGameState state;
    protected Random rnd;
    protected GeminiMCTSParams params;

    protected Map<AbstractAction, GeminiMCTSTreeNode> children = new HashMap<>();

    protected double totValue = 0;
    protected int nVisits = 0;
    protected int fmCallsCount = 0;
    protected int depth;

    public GeminiMCTSTreeNode(GeminiMCTSPlayer player, GeminiMCTSTreeNode parent, AbstractGameState gameState, Random rnd) {
        this.player = player;
        this.parent = parent;
        this.root = parent == null ? this : parent.root;
        this.state = gameState;
        this.rnd = rnd;
        this.params = player.getParameters();
        this.depth = parent == null ? 0 : parent.depth + 1;
        this.fmCallsCount = 0;

        // Initialize children map with available actions
        if (gameState.isNotTerminal()) {
            for (AbstractAction action : player.getForwardModel().computeAvailableActions(state, player.getParameters().actionSpace)) {
                children.put(action, null);
            }
        }
    }

    /**
     * Performs full MCTS search, using the defined budget limits.
     */
    public void mctsSearch() {
        // Variables for tracking time budget
        double avgTimeTaken;
        double acumTimeTaken = 0;
        long remaining;
        int remainingLimit = params.breakMS;
        ElapsedCpuTimer elapsedTimer = new ElapsedCpuTimer();
        if (params.budgetType == BUDGET_TIME) {
            elapsedTimer.setMaxTimeMillis(params.budget);
        }

        // Tracking number of iterations
        int numIters = 0;
        boolean stop = false;

        while (!stop) {
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();

            // 1. Selection + Expansion
            GeminiMCTSTreeNode selected = treePolicy();

            // 2. Simulation (Rollout) - THIS IS WHERE GEMINI'S SMART LOGIC IS
            double delta = selected.rollOut();

            // 3. Backpropagation
            selected.backUp(delta);

            numIters++;

            // Check stopping condition
            PlayerConstants budgetType = params.budgetType;
            if (budgetType == BUDGET_TIME) {
                acumTimeTaken += (elapsedTimerIteration.elapsedMillis());
                avgTimeTaken = acumTimeTaken / numIters;
                remaining = elapsedTimer.remainingTimeMillis();
                stop = remaining <= 2 * avgTimeTaken || remaining <= remainingLimit;
            } else if (budgetType == BUDGET_ITERATIONS) {
                stop = numIters >= params.budget;
            } else if (budgetType == BUDGET_FM_CALLS) {
                stop = fmCallsCount > params.budget;
            }
        }
    }

    /**
     * Selection + expansion steps.
     */
    private GeminiMCTSTreeNode treePolicy() {
        GeminiMCTSTreeNode cur = this;

        while (cur.state.isNotTerminal() && cur.depth < params.maxTreeDepth) {
            List<AbstractAction> unexpanded = cur.unexpandedActions();
            if (!unexpanded.isEmpty()) {
                return cur.expand();
            } else {
                AbstractAction actionChosen = cur.ucb();
                cur = cur.children.get(actionChosen);
            }
        }
        return cur;
    }

    /**
     * Get list of unexpanded actions.
     */
    private List<AbstractAction> unexpandedActions() {
        List<AbstractAction> unexpanded = new ArrayList<>();
        for (AbstractAction action : children.keySet()) {
            if (children.get(action) == null) {
                unexpanded.add(action);
            }
        }
        return unexpanded;
    }

    /**
     * Expand a random unexpanded child.
     */
    private GeminiMCTSTreeNode expand() {
        List<AbstractAction> notChosen = unexpandedActions();
        AbstractAction chosen = notChosen.get(rnd.nextInt(notChosen.size()));

        // Copy state and advance
        AbstractGameState nextState = state.copy();
        advance(nextState, chosen.copy());

        // Create new node
        GeminiMCTSTreeNode tn = new GeminiMCTSTreeNode(player, this, nextState, rnd);
        children.put(chosen, tn);
        return tn;
    }

    /**
     * Advance game state with action.
     */
    private void advance(AbstractGameState gs, AbstractAction act) {
        player.getForwardModel().next(gs, act);
        root.fmCallsCount++;
    }

    /**
     * UCB selection.
     * --- THIS CONTAINS THE FIX ---
     */
    private AbstractAction ucb() {
        AbstractAction bestAction = null;
        double bestValue = -Double.MAX_VALUE;

        for (AbstractAction action : children.keySet()) {
            GeminiMCTSTreeNode child = children.get(action);
            if (child == null)
                throw new AssertionError("Should not be here");

            if (bestAction == null)
                bestAction = action;

            double childValue = child.totValue / (child.nVisits + params.epsilon);
            double explorationTerm = params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));

            // *** START OF FIX ***
            // We are in a 3-player game, so we always maximize our own score.
            // The heuristic value is already from our (the root player's) perspective.
            // We remove the (iAmMoving ? childValue : -childValue) negamax logic.
            double uctValue = childValue + explorationTerm;
            // *** END OF FIX ***

            uctValue = noise(uctValue, params.epsilon, rnd.nextDouble());

            if (uctValue > bestValue) {
                bestAction = action;
                bestValue = uctValue;
            }
        }

        if (bestAction == null)
            throw new AssertionError("We have a null value in UCT");

        root.fmCallsCount++;
        return bestAction;
    }

    /**
     * Perform a Monte Carlo rollout from this node.
     * * --- THIS IS THE KEY "GEMINI" LOGIC ---
     * When an opponent's hand is visible (isHandKnown), we pick the best action
     * for them using the heuristic, but *only for the first step* of the rollout
     * to keep simulations fast.
     */
    private double rollOut() {
        int rolloutDepth = 0;
        AbstractGameState rolloutState = this.state.copy();

        while (!finishRollout(rolloutState, rolloutDepth)) {
            List<AbstractAction> availableActions = player.getForwardModel().computeAvailableActions(rolloutState, player.getParameters().actionSpace);
            if (availableActions.isEmpty()) break;

            AbstractAction chosenAction;

            // --- GEMINI'S SMART SIMULATION LOGIC ---
            int rootPlayer = player.getPlayerID();
            int currentPlayer = rolloutState.getCurrentPlayer();

            if (rolloutState instanceof SGGameState) {
                SGGameState sgState = (SGGameState) rolloutState;

                // Your correct optimization: only run smart logic at the first step
                if (sgState.isHandKnown(rootPlayer, currentPlayer) && rolloutDepth == 0) {
                    chosenAction = getBestAction(sgState, availableActions, currentPlayer);
                } else {
                    // Fallback to random for speed
                    chosenAction = availableActions.get(rnd.nextInt(availableActions.size()));
                }
            } else {
                // Fallback to random if not an SGGameState
                chosenAction = availableActions.get(rnd.nextInt(availableActions.size()));
            }
            // --- END GEMINI LOGIC ---

            advance(rolloutState, chosenAction);
            rolloutDepth++;
        }

        // Evaluate final state
        IStateHeuristic heuristic = params.getStateHeuristic();
        double value = heuristic.evaluateState(rolloutState, player.getPlayerID());

        if (Double.isNaN(value))
            throw new AssertionError("Illegal heuristic value - should be a number");

        return value;
    }

    /**
     * Helper: finds the best action for a player from a known hand
     * by doing a 1-ply lookahead with the heuristic.
     */
    private AbstractAction getBestAction(SGGameState state, List<AbstractAction> actions, int playerToEvaluate) {
        AbstractAction bestAction = null;
        double bestValue = Double.NEGATIVE_INFINITY;
        IStateHeuristic heuristic = params.getStateHeuristic();

        for (AbstractAction action : actions) {
            AbstractGameState nextState = state.copy();
            player.getForwardModel().next(nextState, action); // Note: This doesn't count FM calls for the root, which is fine

            double value = heuristic.evaluateState(nextState, playerToEvaluate);

            if (value > bestValue) {
                bestValue = value;
                bestAction = action;
            }
        }

        if (bestAction == null && !actions.isEmpty()) {
            return actions.get(rnd.nextInt(actions.size()));
        }

        return bestAction;
    }

    /**
     * Check if rollout should finish.
     */
    private boolean finishRollout(AbstractGameState rollerState, int depth) {
        if (depth >= params.rolloutLength)
            return true;
        return !rollerState.isNotTerminal();
    }

    /**
     * Backpropagate value up the tree.
     */
    private void backUp(double result) {
        GeminiMCTSTreeNode n = this;
        while (n != null) {
            n.nVisits++;
            n.totValue += result;
            n = n.parent;
        }
    }

    /**
     * Returns the best action from the root (most visited child).
     */
    public AbstractAction bestAction() {
        double bestValue = -Double.MAX_VALUE;
        AbstractAction bestAction = null;

        for (AbstractAction action : children.keySet()) {
            if (children.get(action) != null) {
                GeminiMCTSTreeNode node = children.get(action);
                double childValue = node.nVisits;
                childValue = noise(childValue, params.epsilon, rnd.nextDouble());

                if (childValue > bestValue) {
                    bestValue = childValue;
                    bestAction = action;
                }
            }
        }

        if (bestAction == null) {
            // Fallback in case no actions are expanded (e.g., extremely low budget)
            if (children.isEmpty()) {
                List<AbstractAction> actions = player.getForwardModel().computeAvailableActions(state, player.getParameters().actionSpace);
                return actions.get(rnd.nextInt(actions.size()));
            }
            // Or pick randomly from available children
            List<AbstractAction> childActions = new ArrayList<>(children.keySet());
            return childActions.get(rnd.nextInt(childActions.size()));
        }

        return bestAction;
    }
}