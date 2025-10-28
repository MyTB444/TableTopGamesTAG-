package players.improvedBasisMCTS;

import core.AbstractGameState;
import core.actions.AbstractAction;
import players.PlayerConstants;
import players.simple.RandomPlayer;
import utilities.ElapsedCpuTimer;

import java.util.*;

import static java.util.stream.Collectors.toList;
import static players.PlayerConstants.*;
import static utilities.Utils.noise;

/**
 * FIXED version of ImprovedBasicTreeNode without the performance bugs
 */
class ImprovedBasicTreeNode {
    // Root node of tree
    ImprovedBasicTreeNode root;
    // Parent of this node
    ImprovedBasicTreeNode parent;
    // Children of this node
    Map<AbstractAction, ImprovedBasicTreeNode> children = new HashMap<>();
    // Depth of this node
    final int depth;

    // Total value of this node
    private double totValue;
    // Number of visits
    private int nVisits;

    // RAVE statistics
    private Map<AbstractAction, Double> raveValues = new HashMap<>();
    private Map<AbstractAction, Integer> raveVisits = new HashMap<>();

    // UCB1-Tuned variance tracking
    private double sumSquares;

    // Number of FM calls and State copies up until this node
    private int fmCallsCount;
    // Parameters guiding the search
    private ImprovedBasicMCTSPlayer player;
    private Random rnd;
    private RandomPlayer randomPlayer = new RandomPlayer();

    // State in this node
    private AbstractGameState state;

    protected ImprovedBasicTreeNode(ImprovedBasicMCTSPlayer player, ImprovedBasicTreeNode parent,
                            AbstractGameState state, Random rnd) {
        this.player = player;
        this.fmCallsCount = 0;
        this.parent = parent;
        this.root = parent == null ? this : parent.root;
        totValue = 0.0;
        sumSquares = 0.0;
        setState(state);
        if (parent != null) {
            depth = parent.depth + 1;
        } else {
            depth = 0;
        }
        this.rnd = rnd;
        randomPlayer.setForwardModel(player.getForwardModel());
    }

    /**
     * Performs full MCTS search, using the defined budget limits.
     */
    void mctsSearch() {
        ImprovedBasicMCTSParams params = player.getParameters();

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

            // Selection + expansion
            ImprovedBasicTreeNode selected = treePolicy();

            // Monte carlo rollout
            double value = selected.rollOut();

            // Back up the value
            selected.backUp(value);

            // Finished iteration
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
     * Selection + expansion with FIXED progressive widening
     */
    private ImprovedBasicTreeNode treePolicy() {
        ImprovedBasicTreeNode cur = this;

        while (cur.state.isNotTerminal() && cur.depth < player.getParameters().maxTreeDepth) {
            if (cur.notFullyExpanded()) {
                // Expand a new node
                return cur.expand();
            } else {
                // Select best child
                cur = cur.selectChild();
            }
        }
        return cur;
    }

    /**
     * Check if node is not fully expanded
     * FIXED: More permissive progressive widening
     */
    private boolean notFullyExpanded() {
        ImprovedBasicMCTSParams params = player.getParameters();

        // Count actual expanded children (non-null)
        long expandedCount = children.values().stream().filter(Objects::nonNull).count();

        // If all possible actions have been tried, we're fully expanded
        if (expandedCount >= children.size()) {
            return false;
        }

        // If not using progressive widening, always allow expansion
        if (!params.useProgressiveWidening) {
            return true;
        }

        // FIXED: More permissive formula
        // Ensure we explore at least 50% of actions or 5 actions minimum
        double minActions = Math.min(children.size() * 0.5, Math.max(5, children.size() * 0.3));

        // Progressive widening formula
        double k_n = params.progressiveWideningC * Math.pow(nVisits + 1, params.progressiveWideningAlpha);

        // Allow expansion if below both thresholds
        return expandedCount < Math.max(minActions, k_n);
    }

    /**
     * Expand a new child
     */
    private ImprovedBasicTreeNode expand() {
        // Get unexpanded actions
        List<AbstractAction> unexpanded = children.entrySet().stream()
                .filter(e -> e.getValue() == null)
                .map(Map.Entry::getKey)
                .collect(toList());

        if (unexpanded.isEmpty()) {
            throw new AssertionError("Cannot expand fully expanded node");
        }

        // Choose random unexpanded action
        AbstractAction chosen = unexpanded.get(rnd.nextInt(unexpanded.size()));

        // Create child state
        AbstractGameState nextState = state.copy();
        advance(nextState, chosen);

        // Create child node
        ImprovedBasicTreeNode child = new ImprovedBasicTreeNode(player, this, nextState, rnd);
        children.put(chosen, child);

        return child;
    }

    /**
     * Select best child using UCB
     */
    private ImprovedBasicTreeNode selectChild() {
        ImprovedBasicTreeNode bestChild = null;
        double bestValue = -Double.MAX_VALUE;
        ImprovedBasicMCTSParams params = player.getParameters();

        for (Map.Entry<AbstractAction, ImprovedBasicTreeNode> entry : children.entrySet()) {
            ImprovedBasicTreeNode child = entry.getValue();
            if (child == null) continue;  // Skip unexpanded

            double uctValue = calculateUCB(entry.getKey(), child);
            uctValue = noise(uctValue, params.epsilon, rnd.nextDouble());

            if (uctValue > bestValue) {
                bestValue = uctValue;
                bestChild = child;
            }
        }

        if (bestChild == null) {
            throw new AssertionError("No child selected!");
        }

        return bestChild;
    }

    /**
     * Calculate UCB value - SIMPLIFIED AND FIXED
     */
    private double calculateUCB(AbstractAction action, ImprovedBasicTreeNode child) {
        ImprovedBasicMCTSParams params = player.getParameters();

        // Basic exploitation value
        double exploitation = child.totValue / (child.nVisits + params.epsilon);

        // Basic exploration term
        double exploration = params.K * Math.sqrt(
                Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));

        // FIXED: Proper UCB1-Tuned with bounds checking
        if (params.useUCB1Tuned && child.nVisits > 1) {
            double mean = child.totValue / child.nVisits;
            double variance = (child.sumSquares / child.nVisits) - (mean * mean);
            variance = Math.max(0, variance);  // Ensure non-negative

            double v_i = variance + Math.sqrt(2 * Math.log(this.nVisits) / child.nVisits);
            exploration = params.K * Math.sqrt(
                    Math.log(this.nVisits) / child.nVisits * Math.min(0.25, v_i));
        }

        // SIMPLIFIED RAVE - only if we have significant visits
        double finalValue = exploitation;
        if (params.useRAVE && raveVisits.getOrDefault(action, 0) > 5) {
            double raveValue = raveValues.getOrDefault(action, 0.0) /
                    raveVisits.getOrDefault(action, 1);
            double beta = Math.sqrt(params.raveK / (3 * this.nVisits + params.raveK));
            finalValue = (1 - beta) * exploitation + beta * raveValue;
        }

        // Handle player perspective
        boolean iAmMoving = state.getCurrentPlayer() == player.getPlayerID();
        if (!iAmMoving) {
            finalValue = -finalValue;
        }

        return finalValue + exploration;
    }

    /**
     * Set state - SIMPLIFIED without expensive ordering
     */
    private void setState(AbstractGameState newState) {
        state = newState;
        if (newState.isNotTerminal()) {
            List<AbstractAction> actions = player.getForwardModel()
                    .computeAvailableActions(state, player.getParameters().actionSpace);

            // Initialize children without expensive ordering
            for (AbstractAction action : actions) {
                children.put(action, null);
                if (player.getParameters().useRAVE) {
                    raveValues.put(action, 0.0);
                    raveVisits.put(action, 0);
                }
            }
        }
    }

    /**
     * Advance state
     */
    private void advance(AbstractGameState gs, AbstractAction action) {
        player.getForwardModel().next(gs, action);
        root.fmCallsCount++;
    }

    /**
     * Perform rollout - SIMPLIFIED
     */
    private double rollOut() {
        int rolloutDepth = 0;
        AbstractGameState rolloutState = state.copy();

        if (player.getParameters().rolloutLength > 0) {
            while (!finishRollout(rolloutState, rolloutDepth)) {
                List<AbstractAction> availableActions = randomPlayer.getForwardModel()
                        .computeAvailableActions(rolloutState, randomPlayer.parameters.actionSpace);
                AbstractAction next = availableActions.get(rnd.nextInt(availableActions.size()));
                advance(rolloutState, next);
                rolloutDepth++;
            }
        }

        // Evaluate final state
        double value = player.getParameters().getStateHeuristic()
                .evaluateState(rolloutState, player.getPlayerID());

        if (Double.isNaN(value)) {
            throw new AssertionError("NaN heuristic value");
        }

        return value;
    }

    /**
     * Check if rollout is finished
     */
    private boolean finishRollout(AbstractGameState rollerState, int depth) {
        if (depth >= player.getParameters().rolloutLength) {
            return true;
        }
        return !rollerState.isNotTerminal();
    }

    /**
     * Back up value through tree
     */
    private void backUp(double value) {
        ImprovedBasicTreeNode n = this;
        while (n != null) {
            n.nVisits++;
            n.totValue += value;
            n.sumSquares += value * value;

            // Update RAVE for all actions (simplified)
            if (n.parent != null && player.getParameters().useRAVE) {
                for (AbstractAction action : n.parent.children.keySet()) {
                    n.parent.raveValues.merge(action, value, Double::sum);
                    n.parent.raveVisits.merge(action, 1, Integer::sum);
                }
            }

            n = n.parent;
        }
    }

    /**
     * Select best action from root
     */
    AbstractAction bestAction() {
        double bestValue = -Double.MAX_VALUE;
        AbstractAction bestAction = null;
        ImprovedBasicMCTSParams params = player.getParameters();

        for (Map.Entry<AbstractAction, ImprovedBasicTreeNode> entry : children.entrySet()) {
            ImprovedBasicTreeNode child = entry.getValue();
            if (child != null) {
                // Primary criterion: visit count
                double childValue = child.nVisits;

                // Tie-breaker: average value
                if (params.useValueTieBreaking) {
                    childValue += (child.totValue / (child.nVisits + params.epsilon)) * 0.01;
                }

                // Add noise
                childValue = noise(childValue, params.epsilon, rnd.nextDouble());

                if (childValue > bestValue) {
                    bestValue = childValue;
                    bestAction = entry.getKey();
                }
            }
        }

        if (bestAction == null) {
            throw new AssertionError("No best action found!");
        }

        return bestAction;
    }
}