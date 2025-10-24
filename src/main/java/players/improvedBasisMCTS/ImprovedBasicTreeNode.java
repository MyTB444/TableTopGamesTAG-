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

    // RAVE statistics: track action values across different states
    private Map<AbstractAction, Double> raveValues = new HashMap<>();
    private Map<AbstractAction, Integer> raveVisits = new HashMap<>();

    // UCB1-Tuned variance tracking
    private double sumSquares;  // Sum of squared rewards

    // Number of FM calls and State copies up until this node
    private int fmCallsCount;
    // Parameters guiding the search
    private ImprovedBasicMCTSPlayer player;
    private Random rnd;
    private RandomPlayer randomPlayer = new RandomPlayer();

    // State in this node (closed loop)
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

        // Tracking number of iterations for iteration budget
        int numIters = 0;

        boolean stop = false;

        while (!stop) {
            // New timer for this iteration
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();

            // Selection + expansion: navigate tree until a node not fully expanded is found, add a new node to the tree
            ImprovedBasicTreeNode selected = treePolicy();

            // Monte carlo rollout: return value of MC rollout from the newly added node
            RolloutResult result = selected.rollOut();

            // Back up the value of the rollout through the tree (with RAVE if enabled)
            selected.backUp(result);

            // Finished iteration
            numIters++;

            // Check stopping condition
            PlayerConstants budgetType = params.budgetType;
            if (budgetType == BUDGET_TIME) {
                // Time budget
                acumTimeTaken += (elapsedTimerIteration.elapsedMillis());
                avgTimeTaken = acumTimeTaken / numIters;
                remaining = elapsedTimer.remainingTimeMillis();
                stop = remaining <= 2 * avgTimeTaken || remaining <= remainingLimit;
            } else if (budgetType == BUDGET_ITERATIONS) {
                // Iteration budget
                stop = numIters >= params.budget;
            } else if (budgetType == BUDGET_FM_CALLS) {
                // FM calls budget
                stop = fmCallsCount > params.budget;
            }
        }
    }

    /**
     * Selection + expansion steps with progressive widening support.
     * - Tree is traversed until a node not fully expanded is found.
     * - A new child of this node is added to the tree.
     *
     * @return - new node added to the tree.
     */
    private ImprovedBasicTreeNode treePolicy() {

        ImprovedBasicTreeNode cur = this;

        // Keep iterating while the state reached is not terminal and the depth of the tree is not exceeded
        while (cur.state.isNotTerminal() && cur.depth < player.getParameters().maxTreeDepth) {
            List<AbstractAction> unexpanded = cur.unexpandedActions();

            if (unexpanded.isEmpty()) {
                // All actions expanded, select best child
                AbstractAction actionChosen = cur.selectAction();
                cur = cur.children.get(actionChosen);
            } else if (cur.shouldExpandMore()) {
                // We should expand a new action
                cur = cur.expand();
                return cur;
            } else {
                // Progressive widening says don't expand more
                // But we have unexpanded actions, so we have expanded children to select from
                // Select from already expanded children
                AbstractAction actionChosen = cur.selectAction();
                cur = cur.children.get(actionChosen);
            }
        }

        return cur;
    }

    /**
     * Check if we should expand more children based on progressive widening.
     */
    private boolean shouldExpandMore() {
        List<AbstractAction> unexpanded = unexpandedActions();
        if (unexpanded.isEmpty()) {
            return false;  // All actions expanded
        }

        ImprovedBasicMCTSParams params = player.getParameters();
        if (!params.useProgressiveWidening) {
            return true;  // Always expand if not using progressive widening
        }

        // Progressive widening: k(n) = C * n^alpha
        // We can expand more children if current_children < k(n)
        int expandedChildren = children.size() - unexpanded.size();

        // Always expand at least one child
        if (expandedChildren == 0) {
            return true;
        }

        double k_n = params.progressiveWideningC * Math.pow(nVisits, params.progressiveWideningAlpha);

        return expandedChildren < k_n;
    }

    private void setState(AbstractGameState newState) {
        state = newState;
        if (newState.isNotTerminal()) {
            List<AbstractAction> actions = player.getForwardModel()
                    .computeAvailableActions(state, player.getParameters().actionSpace);

            // Sort actions by heuristic value for better expansion order
            actions = orderActionsByHeuristic(actions);

            for (AbstractAction action : actions) {
                children.put(action, null); // mark a new node to be expanded
                // Initialize RAVE stats
                raveValues.put(action, 0.0);
                raveVisits.put(action, 0);
            }
        }
    }

    /**
     * Order actions by quick heuristic evaluation for better expansion order.
     */
    private List<AbstractAction> orderActionsByHeuristic(List<AbstractAction> actions) {
        ImprovedBasicMCTSParams params = player.getParameters();
        if (!params.useActionOrdering || actions.size() < 5) {
            return actions;  // Don't bother for small action spaces
        }

        // Quick evaluation of each action
        List<ActionValue> scoredActions = new ArrayList<>();
        for (AbstractAction action : actions) {
            AbstractGameState testState = state.copy();
            player.getForwardModel().next(testState, action.copy());
            double value = params.getStateHeuristic().evaluateState(testState, player.getPlayerID());
            scoredActions.add(new ActionValue(action, value));
        }

        // Sort descending by value
        scoredActions.sort((a, b) -> Double.compare(b.value, a.value));

        return scoredActions.stream().map(av -> av.action).collect(toList());
    }

    /**
     * Helper class for action ordering.
     */
    private static class ActionValue {
        AbstractAction action;
        double value;

        ActionValue(AbstractAction action, double value) {
            this.action = action;
            this.value = value;
        }
    }

    /**
     * @return A list of the unexpanded Actions from this State
     */
    private List<AbstractAction> unexpandedActions() {
        return children.keySet().stream().filter(a -> children.get(a) == null).collect(toList());
    }

    /**
     * Expands the node by creating a new child node and adding to the tree.
     * Uses heuristic ordering to expand promising actions first.
     *
     * @return - new child node.
     */
    private ImprovedBasicTreeNode expand() {
        // Pick best unexpanded action (they're already ordered by heuristic)
        List<AbstractAction> notChosen = unexpandedActions();
        AbstractAction chosen = notChosen.get(0);  // Take the first (best) unexpanded action

        // copy the current state and advance it using the chosen action
        AbstractGameState nextState = state.copy();
        advance(nextState, chosen.copy());

        // then instantiate a new node
        ImprovedBasicTreeNode tn = new ImprovedBasicTreeNode(player, this, nextState, rnd);
        children.put(chosen, tn);
        return tn;
    }

    /**
     * Advance the current game state with the given action, count the FM call.
     *
     * @param gs  - current game state
     * @param act - action to apply
     */
    private void advance(AbstractGameState gs, AbstractAction act) {
        player.getForwardModel().next(gs, act);
        root.fmCallsCount++;
    }

    /**
     * Enhanced selection using UCB with optional RAVE and UCB1-Tuned.
     */
    private AbstractAction selectAction() {
        AbstractAction bestAction = null;
        double bestValue = -Double.MAX_VALUE;
        ImprovedBasicMCTSParams params = player.getParameters();

        for (AbstractAction action : children.keySet()) {
            ImprovedBasicTreeNode child = children.get(action);
            // Skip unexpanded children (can happen with progressive widening)
            if (child == null)
                continue;

            if (bestAction == null)
                bestAction = action;

            double uctValue = calculateUCTValue(action, child);

            // Apply small noise to break ties randomly
            uctValue = noise(uctValue, params.epsilon, player.getRnd().nextDouble());

            // Assign value
            if (uctValue > bestValue) {
                bestAction = action;
                bestValue = uctValue;
            }
        }

        if (bestAction == null)
            throw new AssertionError("We have a null value in UCT : shouldn't really happen!");

        root.fmCallsCount++;  // log one iteration complete
        return bestAction;
    }

    /**
     * Calculate UCT value with optional RAVE and UCB1-Tuned enhancements.
     */
    private double calculateUCTValue(AbstractAction action, ImprovedBasicTreeNode child) {
        ImprovedBasicMCTSParams params = player.getParameters();

        // Standard exploitation value
        double hvVal = child.totValue;
        double childValue = hvVal / (child.nVisits + params.epsilon);

        // Standard exploration term
        double explorationTerm = params.K * Math.sqrt(
                Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));

        // UCB1-Tuned: add variance term
        if (params.useUCB1Tuned && child.nVisits > 1) {
            double variance = (child.sumSquares / child.nVisits) - (childValue * childValue);
            double v_i = variance + Math.sqrt(2 * Math.log(this.nVisits) / child.nVisits);
            explorationTerm = params.K * Math.sqrt(
                    Math.log(this.nVisits) / child.nVisits * Math.min(0.25, v_i));
        }

        double uctValue;
        boolean iAmMoving = state.getCurrentPlayer() == player.getPlayerID();

        if (params.useRAVE && raveVisits.get(action) > 0) {
            // RAVE enhancement: blend MC value with RAVE value
            double raveValue = raveValues.get(action) / (raveVisits.get(action) + params.epsilon);

            // RAVE weight decreases as we get more MC samples
            double beta = Math.sqrt(params.raveK / (3 * this.nVisits + params.raveK));

            // Blend MC and RAVE values
            double blendedValue = (1 - beta) * childValue + beta * raveValue;
            uctValue = iAmMoving ? blendedValue : -blendedValue;
        } else {
            uctValue = iAmMoving ? childValue : -childValue;
        }

        uctValue += explorationTerm;

        return uctValue;
    }

    /**
     * Perform a Monte Carlo rollout from this node.
     * Returns both the value and the actions taken for RAVE updates.
     *
     * @return - rollout result with value and actions taken.
     */
    private RolloutResult rollOut() {
        int rolloutDepth = 0;
        List<AbstractAction> actionsInRollout = new ArrayList<>();

        // If rollouts are enabled, select actions for the rollout
        AbstractGameState rolloutState = state.copy();
        if (player.getParameters().rolloutLength > 0) {
            while (!finishRollout(rolloutState, rolloutDepth)) {
                List<AbstractAction> availableActions = randomPlayer.getForwardModel()
                        .computeAvailableActions(rolloutState, randomPlayer.parameters.actionSpace);
                AbstractAction next = randomPlayer.getAction(rolloutState, availableActions);
                actionsInRollout.add(next);
                advance(rolloutState, next);
                rolloutDepth++;
            }
        }

        // Evaluate final state
        double value = player.getParameters().getStateHeuristic()
                .evaluateState(rolloutState, player.getPlayerID());

        if (Double.isNaN(value))
            throw new AssertionError("Illegal heuristic value - should be a number");

        return new RolloutResult(value, actionsInRollout);
    }

    /**
     * Helper class to store rollout results.
     */
    private static class RolloutResult {
        double value;
        List<AbstractAction> actions;

        RolloutResult(double value, List<AbstractAction> actions) {
            this.value = value;
            this.actions = actions;
        }
    }

    /**
     * Checks if rollout is finished.
     */
    private boolean finishRollout(AbstractGameState rollerState, int depth) {
        if (depth >= player.getParameters().rolloutLength)
            return true;
        return !rollerState.isNotTerminal();
    }

    /**
     * Back up the value through the tree with RAVE updates.
     *
     * @param result - rollout result to backup
     */
    private void backUp(RolloutResult result) {
        ImprovedBasicTreeNode n = this;
        while (n != null) {
            n.nVisits++;
            n.totValue += result.value;
            n.sumSquares += result.value * result.value;

            // Update RAVE statistics
            if (player.getParameters().useRAVE && n.parent != null) {
                for (AbstractAction action : result.actions) {
                    // Update RAVE for all actions that could have been taken
                    if (n.parent.children.containsKey(action)) {
                        n.parent.updateRAVE(action, result.value);
                    }
                }
            }

            n = n.parent;
        }
    }

    /**
     * Update RAVE statistics for an action.
     */
    private void updateRAVE(AbstractAction action, double value) {
        if (raveValues.containsKey(action)) {
            raveValues.put(action, raveValues.get(action) + value);
            raveVisits.put(action, raveVisits.get(action) + 1);
        }
    }

    /**
     * Calculates the best action from the root according to the most visited node
     * (with optional value tie-breaking).
     *
     * @return - the best AbstractAction
     */
    AbstractAction bestAction() {
        double bestValue = -Double.MAX_VALUE;
        AbstractAction bestAction = null;
        ImprovedBasicMCTSParams params = player.getParameters();

        for (AbstractAction action : children.keySet()) {
            if (children.get(action) != null) {
                ImprovedBasicTreeNode node = children.get(action);

                // Primary criterion: visit count
                double childValue = node.nVisits;

                // Tie-breaker: average value (for equally visited nodes)
                if (params.useValueTieBreaking) {
                    childValue += (node.totValue / (node.nVisits + params.epsilon)) * 0.01;
                }

                // Apply small noise to break ties randomly
                childValue = noise(childValue, params.epsilon, player.getRnd().nextDouble());

                // Save best value
                if (childValue > bestValue) {
                    bestValue = childValue;
                    bestAction = action;
                }
            }
        }

        if (bestAction == null) {
            throw new AssertionError("Unexpected - no selection made.");
        }

        return bestAction;
    }
}