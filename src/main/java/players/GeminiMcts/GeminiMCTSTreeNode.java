package players.GeminiMcts; // Updated package

import core.AbstractGameState;
import core.actions.AbstractAction;
import core.interfaces.IStateHeuristic;
import games.sushigo.SGGameState; // Make sure this import path is correct
import players.PlayerConstants;
import utilities.ElapsedCpuTimer;

// --- IMPORTS FOR NEW FAST ROLLOUT ---
// *** PLEASE CHECK THESE PATHS ***
// You may need to change these imports to match your project structure.
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
        // Optional: Add your debug line here to check iteration count
        // System.out.println(player.toString() + " completed " + numIters + " iterations.");
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

                    // *** USE THE NEW FAST-ACTION PICKER ***
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
     * Helper: finds the best action for a player from a known hand.
     * --- STATE-AWARE FAST HEURISTIC ---
     * Checks player's board to make context-aware decisions (Tempura pairs, Sashimi triples, Wasabi synergy)
     */
    private AbstractAction getBestAction(SGGameState state, List<AbstractAction> actions, int playerToEvaluate) {
        AbstractAction bestAction = null;
        double bestValue = Double.NEGATIVE_INFINITY;

        // Get counts of key cards the player has already played THIS ROUND
        int tempuraCount = state.getPlayedCardTypes(SGCard.SGCardType.Tempura, playerToEvaluate).getValue();
        int sashimiCount = state.getPlayedCardTypes(SGCard.SGCardType.Sashimi, playerToEvaluate).getValue();
        int wasabiCount = state.getPlayedCardTypes(SGCard.SGCardType.Wasabi, playerToEvaluate).getValue();

        // Check if there's an unused Wasabi on the board
        // Wasabi stays until used, so if wasabi > 0, we can use it
        boolean hasUnusedWasabi = wasabiCount > 0;

        for (AbstractAction action : actions) {

            if (action instanceof ChooseCard) {
                ChooseCard chooseCard = (ChooseCard) action;
                SGCard card = (SGCard) chooseCard.getCard(state);

                double value = 0;

                // Context-aware card evaluation
                switch (card.type) {
                    case SquidNigiri:
                        value = hasUnusedWasabi ? 9.0 : 3.0; // 3x with Wasabi!
                        break;
                    case SalmonNigiri:
                        value = hasUnusedWasabi ? 6.0 : 2.0;
                        break;
                    case EggNigiri:
                        value = hasUnusedWasabi ? 3.0 : 1.0;
                        break;
                    case Maki:
                        value = card.count * 1.5; // 1.5, 3.0, or 4.5 based on maki roll count
                        break;
                    case Pudding:
                        value = 1.0;
                        break;
                    case Tempura:
                        // HUGE value if it completes a pair (odd count means we need 1 more)
                        value = (tempuraCount % 2 == 1) ? 5.0 : 2.5;
                        break;
                    case Sashimi:
                        // HUGE value if it completes a triple (count=2 means we need 1 more)
                        value = (sashimiCount % 3 == 2) ? 10.0 : 3.0;
                        break;
                    case Dumpling:
                        value = 1.5;
                        break;
                    case Wasabi:
                        value = -0.1; // Bad on its own
                        break;
                    case Chopsticks:
                        value = 0.5;
                        break;
                    default:
                        value = 0.5;
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