package players.GeminiMcts;

import core.AbstractGameState;
import core.actions.AbstractAction;
import core.components.Counter;
import core.interfaces.IStateHeuristic;
import games.sushigo.SGGameState;
import players.PlayerConstants;
import utilities.ElapsedCpuTimer;

// --- IMPORTS FOR FAST ROLLOUT HELPER ---
import games.sushigo.actions.ChooseCard;
import games.sushigo.cards.SGCard;
// --- END OF IMPORTS ---

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
        // ... (Time budget code is all correct, no changes) ...
        double avgTimeTaken;
        double acumTimeTaken = 0;
        long remaining;
        int remainingLimit = params.breakMS;
        ElapsedCpuTimer elapsedTimer = new ElapsedCpuTimer();
        if (params.budgetType == BUDGET_TIME) {
            elapsedTimer.setMaxTimeMillis(params.budget);
        }

        int numIters = 0;
        boolean stop = false;

        while (!stop) {
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();

            // 1. Selection + Expansion (NOW THE SMART PART)
            GeminiMCTSTreeNode selected = treePolicy();

            // 2. Simulation (Rollout) (NOW 100% FAST)
            double delta = selected.rollOut();

            // 3. Backpropagation
            selected.backUp(delta);

            numIters++;

            // ... (Stopping condition code is all correct, no changes) ...
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
     * --- THIS IS NOW THE CORE GEMINI LOGIC ---
     */
    private GeminiMCTSTreeNode treePolicy() {
        GeminiMCTSTreeNode cur = this;

        while (cur.state.isNotTerminal() && cur.depth < params.maxTreeDepth) {

            // --- NEW GEMINI LOGIC: Smart Opponent Selection ---
            int rootPlayer = player.getPlayerID();
            int currentPlayer = cur.state.getCurrentPlayer();

            if (currentPlayer != rootPlayer && cur.state instanceof SGGameState) {
                SGGameState sgState = (SGGameState) cur.state;
                if (sgState.isHandKnown(rootPlayer, currentPlayer)) {

                    // This is an opponent with a known hand.
                    // We don't use UCB. We *force* the "best" move for them.
                    List<AbstractAction> availableActions = player.getForwardModel().computeAvailableActions(cur.state, player.getParameters().actionSpace);
                    if (availableActions.isEmpty()) break; // Should not happen, but safeguard

                    // Use our fast helper to predict their move
                    AbstractAction bestOpponentAction = cur.getBestAction(sgState, availableActions, currentPlayer);

                    // If we've never seen this move before, expand it
                    if (cur.children.get(bestOpponentAction) == null) {
                        return cur.expandSpecificAction(bestOpponentAction); // A new helper function
                    } else {
                        // We have seen it, traverse down this "forced" path
                        cur = cur.children.get(bestOpponentAction);
                        continue; // Restart the while loop from the new node
                    }
                }
            }
            // --- END OF NEW LOGIC ---

            // If it's our move, or an opponent with an unknown hand, use standard MCTS
            List<AbstractAction> unexpanded = cur.unexpandedActions();
            if (!unexpanded.isEmpty()) {
                return cur.expand(); // Standard random expansion
            } else {
                AbstractAction actionChosen = cur.ucb(); // Standard UCB selection
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
        return expandSpecificAction(chosen);
    }

    /**
     * New Helper: Expands a *specific* action.
     * Used by both standard expand() and our new smart treePolicy().
     */
    private GeminiMCTSTreeNode expandSpecificAction(AbstractAction chosen) {
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
     * UCB selection. (This contains the 3-player fix)
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

            // Correct 3-player UCB logic
            double uctValue = childValue + explorationTerm;
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
     * --- THIS IS NOW 100% FAST AND RANDOM ---
     */
    private double rollOut() {
        int rolloutDepth = 0;
        AbstractGameState rolloutState = this.state.copy();

        while (!finishRollout(rolloutState, rolloutDepth)) {
            List<AbstractAction> availableActions = player.getForwardModel().computeAvailableActions(rolloutState, player.getParameters().actionSpace);
            if (availableActions.isEmpty()) break;

            // --- REVERTED TO 100% FAST RANDOM ROLLOUT ---
            AbstractAction chosenAction = availableActions.get(rnd.nextInt(availableActions.size()));
            // --- END OF REVERT ---

            advance(rolloutState, chosenAction);
            rolloutDepth++;
        }

        // Evaluate final state (this is where your SGHeuristicGemini is called)
        IStateHeuristic heuristic = params.getStateHeuristic();
        double value = heuristic.evaluateState(rolloutState, player.getPlayerID());

        if (Double.isNaN(value))
            throw new AssertionError("Illegal heuristic value - should be a number");

        return value;
    }

    /**
     * Helper: finds the best action for a player from a known hand.
     * --- THIS IS THE CORRECTED, STATE-AWARE, FAST VERSION ---
     * It now uses the correct API (getPlayedCardTypes) from SGGameState.
     */
    private AbstractAction getBestAction(SGGameState state, List<AbstractAction> actions, int playerToEvaluate) {
        AbstractAction bestAction = null;
        double bestValue = Double.NEGATIVE_INFINITY;

        // --- START OF FIX ---
        // This is the correct way to get the player's board state,
        // based on the logic in SGCard.java.
        // This returns a Map<SGCardType, Counter>.
        Map<SGCard.SGCardType, Counter> playerBoardCounters = state.getPlayedCardTypes()[playerToEvaluate];

        // Get the *values* from the counters. This is much faster (no loop).
        int tempuraCount = playerBoardCounters.get(SGCard.SGCardType.Tempura).getValue();
        int sashimiCount = playerBoardCounters.get(SGCard.SGCardType.Sashimi).getValue();
        boolean hasUnusedWasabi = playerBoardCounters.get(SGCard.SGCardType.Wasabi).getValue() > 0;
        // --- END OF FIX ---

        for (AbstractAction action : actions) {

            // Your existing logic for getting the card was correct
            if (action instanceof games.sushigo.actions.ChooseCard) {
                games.sushigo.actions.ChooseCard chooseCard = (games.sushigo.actions.ChooseCard) action;
                SGCard card = (SGCard) chooseCard.getCard(state);

                double value = 0;

                // This switch statement logic remains the same, but now it's
                // using the *correct* counts for hasUnusedWasabi, tempuraCount, etc.
                switch (card.type) {
                    case SquidNigiri:
                        value = hasUnusedWasabi ? 9.0 : 3.0; // HUGE bonus if we have wasabi
                        break;
                    case SalmonNigiri:
                        value = hasUnusedWasabi ? 6.0 : 2.0;
                        break;
                    case EggNigiri:
                        value = hasUnusedWasabi ? 3.0 : 1.0;
                        break;
                    case Maki:
                        value = card.count * 1.5; // Upped value, Maki is important (1.5, 3.0, 4.5)
                        break;
                    case Pudding:
                        value = 1.0;
                        break;
                    case Tempura:
                        // Value is 5 if it completes a pair, otherwise 2.5
                        value = (tempuraCount % 2 == 1) ? 5.0 : 2.5;
                        break;
                    case Sashimi:
                        // Value is 10 if it completes a set, otherwise 3.0
                        value = (sashimiCount % 3 == 2) ? 10.0 : 3.0; // Upped base value
                        break;
                    case Dumpling:
                        value = 1.5; // Hard to value quickly, 1.5 is fine
                        break;
                    case Wasabi:
                        // Simple logic: it's slightly bad on its own.
                        value = -0.1;
                        break;
                    case Chopsticks:
                        value = 0.5; // Flexibility
                        break;
                    default:
                        value = 0.5; // Unknown card
                        break;
                }

                if (value > bestValue) {
                    bestValue = value;
                    bestAction = action;
                }

            } else {
                // Fallback for non-ChooseCard actions
                if (bestAction == null) {
                    bestAction = action;
                }
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
        // ... (This logic is all correct, no changes) ...
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
            if (children.isEmpty()) {
                List<AbstractAction> actions = player.getForwardModel().computeAvailableActions(state, player.getParameters().actionSpace);
                return actions.get(rnd.nextInt(actions.size()));
            }
            List<AbstractAction> childActions = new ArrayList<>(children.keySet());
            return childActions.get(rnd.nextInt(childActions.size()));
        }

        return bestAction;
    }
}