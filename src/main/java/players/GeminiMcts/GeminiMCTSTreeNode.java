package players.GeminiMcts;

import core.AbstractGameState;
import core.actions.AbstractAction;
import core.components.Counter;
import core.interfaces.IStateHeuristic;
import games.sushigo.SGGameState;
import players.PlayerConstants;
import utilities.ElapsedCpuTimer;

// --- IMPORTS ---
import games.sushigo.actions.ChooseCard;
import games.sushigo.cards.SGCard;
import java.util.*;

import static players.PlayerConstants.*;
import static utilities.Utils.noise;

public class GeminiMCTSTreeNode {

    // --- Core MCTS Fields ---
    protected GeminiMCTSPlayer player;
    protected GeminiMCTSTreeNode parent;
    protected GeminiMCTSTreeNode root;
    protected AbstractGameState state;
    protected Random rnd;
    protected GeminiMCTSParams params;
    protected Map<AbstractAction, GeminiMCTSTreeNode> children = new HashMap<>();

    // --- Statistics ---
    protected double totValue = 0;
    protected int nVisits = 0;
    protected int fmCallsCount = 0;
    protected int depth;

    // --- Gemini Logic Fields ---
    protected AbstractAction cachedOpponentPrediction = null; // Caching logic

    // --- SIMPLE STATIC COUNTERS FOR LOGGING ---
    private static int previousMainRound = -1;
    private static int turnCounterInRound = 0;

    /**
     * --- CONSTRUCTOR ---
     * Writes to cache.
     */
    public GeminiMCTSTreeNode(GeminiMCTSPlayer player, GeminiMCTSTreeNode parent, AbstractGameState gameState, Random rnd) {
        this.player = player;
        this.parent = parent;
        this.root = parent == null ? this : parent.root;
        this.state = gameState;
        this.rnd = rnd;
        this.params = player.getParameters();
        this.depth = parent == null ? 0 : parent.depth + 1;
        this.fmCallsCount = 0;
        this.cachedOpponentPrediction = null; // Default to null

        // Initialize children map and check for caching
        if (gameState.isNotTerminal()) {

            List<AbstractAction> availableActions = player.getForwardModel().computeAvailableActions(state, player.getParameters().actionSpace);

            // --- CACHING LOGIC ---
            int rootPlayer = player.getPlayerID();
            int currentPlayer = gameState.getCurrentPlayer();

            if (currentPlayer != rootPlayer && gameState instanceof SGGameState) {
                SGGameState sgState = (SGGameState) gameState;
                // Predicts at all depths in this version
                if (sgState.isHandKnown(rootPlayer, currentPlayer) && !availableActions.isEmpty()) {
                    this.cachedOpponentPrediction = this.predictOpponentAction(sgState, availableActions, currentPlayer);
                }
            }
            // --- END CACHING LOGIC ---

            for (AbstractAction action : availableActions) {
                children.put(action, null);
            }
        }
    }

    /**
     * --- mctsSearch ---
     * Uses the simple static counter for logging.
     */
    public void mctsSearch() {
        // --- Budget Declarations ---
        double avgTimeTaken = 0;
        double acumTimeTaken = 0;
        long remaining = 0;
        int remainingLimit = params.breakMS;
        ElapsedCpuTimer elapsedTimer = new ElapsedCpuTimer();
        if (params.budgetType == BUDGET_TIME) {
            elapsedTimer.setMaxTimeMillis(params.budget);
        }

        // --- Performance Monitoring Setup ---
        long startTime = System.currentTimeMillis();
        int currentMainRound = -1; // Actual Main Round (1, 2, or 3)

        // --- Update Static Counters ---
        if (state instanceof SGGameState) {
            SGGameState sgState = (SGGameState) state;
            // *** YOU MIGHT NEED TO CHANGE 'getGameRound().getValue()' ***
            try {
                // Replace if needed. Assumes method returns 1, 2, or 3.
                currentMainRound = sgState.getRoundCounter();
            } catch (Exception e) {
                currentMainRound = -1; // Fallback
            }

            // Check if main round changed
            if (currentMainRound != previousMainRound) {
                turnCounterInRound = 0; // Reset turn counter for the new round
                previousMainRound = currentMainRound; // Update the round tracker
            }
        }
        // Increment turn for this specific search call
        turnCounterInRound++;
        int thisTurn = turnCounterInRound; // Capture the turn number for this specific log entry
        // --- End Counter Update ---


        // --- Main Loop ---
        int numIters = 0;
        boolean stop = false;
        while (!stop) {
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
            GeminiMCTSTreeNode selected = treePolicy();
            double delta = selected.rollOut();
            selected.backUp(delta);
            numIters++;

            // --- Stopping Condition ---
            PlayerConstants budgetType = params.budgetType;
            if (budgetType == BUDGET_TIME) {
                acumTimeTaken += (elapsedTimerIteration.elapsedMillis());
                if (numIters > 0) avgTimeTaken = acumTimeTaken / numIters;
                remaining = elapsedTimer.remainingTimeMillis();
                stop = remaining <= 2 * avgTimeTaken || remaining <= remainingLimit;
            } else if (budgetType == BUDGET_ITERATIONS) {
                stop = numIters >= params.budget;
            } else if (budgetType == BUDGET_FM_CALLS) {
                stop = fmCallsCount > params.budget;
            }
        }

        // --- Simple Logging ---
        long elapsed = System.currentTimeMillis() - startTime;
        if (elapsed == 0) elapsed = 1;

/*        System.out.println(String.format(
                "Round %d / Turn %d: %d iterations in %dms (%.1f iter/ms)",
                currentMainRound, // Use the detected main round
                thisTurn,          // Use the captured turn number for this search
                numIters,
                elapsed,
                (double) numIters / elapsed
        ));*/
    }


    /**
     * --- treePolicy() ---
     * Implements "One-Step Prediction" using cache.
     */
    private GeminiMCTSTreeNode treePolicy() {
        GeminiMCTSTreeNode cur = this;

        while (cur.state.isNotTerminal() && cur.depth < params.maxTreeDepth) {

            // --- MODIFIED CACHE-READING LOGIC ---
            if (cur.cachedOpponentPrediction != null) {
                AbstractAction predictedAction = cur.cachedOpponentPrediction;

                if (!cur.children.containsKey(predictedAction)) {
                    // Failsafe: Revert to standard MCTS for this node
                    return standardMCTSStep(cur);
                }

                if (cur.children.get(predictedAction) == null) {
                    // Expand the predicted action
                    return cur.expandSpecificAction(predictedAction);
                } else {
                    // --- KEY CHANGE ---
                    // Traverse the forced path *for one step only*
                    cur = cur.children.get(predictedAction);
                    // *Immediately* proceed to standard MCTS step from the new node
                    return standardMCTSStep(cur); // Use return, not continue
                    // --- END KEY CHANGE ---
                }
            }
            // --- END CACHE-READING LOGIC ---

            // If no cached prediction, fall through to standard MCTS
            return standardMCTSStep(cur);
        }
        return cur;
    }

    /**
     * Helper method for a standard MCTS selection/expansion step
     */
    private GeminiMCTSTreeNode standardMCTSStep(GeminiMCTSTreeNode cur) {
        List<AbstractAction> unexpanded = cur.unexpandedActions();
        if (!unexpanded.isEmpty()) {
            return cur.expand(); // Standard random expansion
        } else {
            AbstractAction actionChosen = cur.ucb(); // Standard UCB selection
            if (actionChosen == null) return cur; // Handles case where all children are terminal or map empty
            return cur.children.get(actionChosen);
        }
    }

    // (unexpandedActions - unchanged)
    private List<AbstractAction> unexpandedActions() {
        List<AbstractAction> unexpanded = new ArrayList<>();
        for (AbstractAction action : children.keySet()) {
            if (children.get(action) == null) {
                unexpanded.add(action);
            }
        }
        return unexpanded;
    }

    // (expand - Standard random expansion)
    private GeminiMCTSTreeNode expand() {
        List<AbstractAction> notChosen = unexpandedActions();
        if (notChosen.isEmpty()) return this;
        AbstractAction chosen = notChosen.get(rnd.nextInt(notChosen.size()));
        return expandSpecificAction(chosen);
    }

    // (expandSpecificAction - unchanged)
    private GeminiMCTSTreeNode expandSpecificAction(AbstractAction chosen) {
        AbstractGameState nextState = state.copy();
        advance(nextState, chosen.copy());
        GeminiMCTSTreeNode tn = new GeminiMCTSTreeNode(player, this, nextState, rnd);
        children.put(chosen, tn);
        return tn;
    }

    // (advance - unchanged)
    private void advance(AbstractGameState gs, AbstractAction act) {
        player.getForwardModel().next(gs, act);
        root.fmCallsCount++;
    }

    /**
     * UCB selection. (Standard UCB, 3-player fixed)
     */
    private AbstractAction ucb() {
        AbstractAction bestAction = null;
        double bestValue = -Double.MAX_VALUE;

        for (AbstractAction action : children.keySet()) {
            GeminiMCTSTreeNode child = children.get(action);
            if (child == null) {
                // Should not happen if standardMCTSStep calls this correctly
                continue; // Ignore unexpanded children
            }

            if (bestAction == null) bestAction = action; // Initialize

            double childValue = child.totValue / (child.nVisits + params.epsilon);
            double explorationTerm = params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));
            double uctValue = childValue + explorationTerm;
            uctValue = noise(uctValue, params.epsilon, rnd.nextDouble());

            if (uctValue > bestValue) {
                bestAction = action;
                bestValue = uctValue;
            }
        }

        if (bestAction == null) {
            // Can happen if children map is empty OR no children have been expanded yet
            return null; // Let standardMCTSStep handle this
        }

        root.fmCallsCount++;
        return bestAction;
    }

    /**
     * Perform a Monte Carlo rollout. (100% FAST RANDOM)
     */
    private double rollOut() {
        int rolloutDepth = 0;
        AbstractGameState rolloutState = this.state.copy();
        while (!finishRollout(rolloutState, rolloutDepth)) {
            List<AbstractAction> availableActions = player.getForwardModel().computeAvailableActions(rolloutState, player.getParameters().actionSpace);
            if (availableActions.isEmpty()) break;
            AbstractAction chosenAction = availableActions.get(rnd.nextInt(availableActions.size()));
            advance(rolloutState, chosenAction);
            rolloutDepth++;
        }
        IStateHeuristic heuristic = params.getStateHeuristic();
        double value = heuristic.evaluateState(rolloutState, player.getPlayerID());
        if (Double.isNaN(value))
            throw new AssertionError("Illegal heuristic value - should be a number");
        return value;
    }

    /**
     * --- predictOpponentAction ---
     * Fast, "selfish" prediction logic.
     */
    private AbstractAction predictOpponentAction(SGGameState state, List<AbstractAction> actions, int opponentID) {
        AbstractAction bestAction = null;
        double bestValue = Double.NEGATIVE_INFINITY;

        Map<SGCard.SGCardType, Counter> oppBoardCounters = state.getPlayedCardTypes()[opponentID];
        int oppTempuraCount = oppBoardCounters.get(SGCard.SGCardType.Tempura).getValue();
        int oppSashimiCount = oppBoardCounters.get(SGCard.SGCardType.Sashimi).getValue();
        boolean oppHasUnusedWasabi = oppBoardCounters.get(SGCard.SGCardType.Wasabi).getValue() > 0;

        for (AbstractAction action : actions) {
            if (action instanceof games.sushigo.actions.ChooseCard) {
                games.sushigo.actions.ChooseCard chooseCard = (games.sushigo.actions.ChooseCard) action;
                SGCard card = (SGCard) chooseCard.getCard(state);

                double value = 0;
                switch (card.type) {
                    case SquidNigiri:   value = oppHasUnusedWasabi ? 9.0 : 3.0; break;
                    case SalmonNigiri:  value = oppHasUnusedWasabi ? 6.0 : 2.0; break;
                    case EggNigiri:     value = oppHasUnusedWasabi ? 3.0 : 1.0; break;
                    case Maki:          value = card.count * 1.5; break;
                    case Pudding:       value = 1.0; break;
                    case Tempura:       value = (oppTempuraCount % 2 == 1) ? 5.0 : 2.5; break;
                    case Sashimi:       value = (oppSashimiCount % 3 == 2) ? 10.0 : 3.0; break;
                    case Dumpling:      value = 1.5; break;
                    case Wasabi:        value = -0.1; break;
                    case Chopsticks:    value = 0.5; break;
                    default:            value = 0.5; break;
                }

                if (value > bestValue) {
                    bestValue = value;
                    bestAction = action;
                }
            } else {
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

    // (finishRollout - unchanged)
    private boolean finishRollout(AbstractGameState rollerState, int depth) {
        if (depth >= params.rolloutLength)
            return true;
        return !rollerState.isNotTerminal();
    }

    // (backUp - unchanged)
    private void backUp(double result) {
        GeminiMCTSTreeNode n = this;
        while (n != null) {
            n.nVisits++;
            n.totValue += result;
            n = n.parent;
        }
    }

    // (bestAction - unchanged)
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