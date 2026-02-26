# VAMP Trajectory Optimization Implementation Plan

## Overview

This document outlines the roadmap for implementing SIMD-accelerated trajectory optimization in VAMP, inspired by CuRobo's approach. The goal is to create a high-performance trajectory optimizer that leverages VAMP's existing SIMD infrastructure and planning capabilities.

## Current Status

### ✅ Completed (Phase 1)

**Core Infrastructure**
- [x] Trajectory representation with time parameterization (`trajectory.hh`)
- [x] Settings and hyperparameters (`trajectory_optimizer_settings.hh`)
- [x] Cost functions (smoothness, acceleration, path length) (`trajectory_cost.hh`)
- [x] Constraint checking (collision, velocity, acceleration, joint limits) (`trajectory_constraints.hh`)
- [x] Gradient descent optimizer (`trajectory_optimizer.hh`)
- [x] Demo program with RRTC seeding (`trajectory_optimizer_example.cc`)
- [x] Successful compilation and execution

**Key Achievements**
- Working end-to-end pipeline: plan → simplify → trajectory optimize
- SIMD-compatible implementation (works with AVX/NEON/WASM)
- Collision-aware optimization using VAMP's existing collision checking
- Execution time: ~10ms for 50 iterations on 17-waypoint trajectory

**Current Limitations**
- Single trajectory optimization (no batch/multi-seed parallelization)
- Simple gradient descent (no quasi-Newton methods)
- Fixed learning rate (no adaptive step size)
- Distance-based time parameterization only
- Sequential gradient computation (not fully SIMD-accelerated)

---

## Development Roadmap

### 🔥 Phase 2: Foundation Improvements (HIGHEST PRIORITY)

These changes will make the current optimizer work significantly better.

#### 2.1 Fix Gradient Descent Issues
**Priority**: 🔴 Critical
**Effort**: Small (1-2 days)
**Impact**: High - Enables actual trajectory improvements

**Tasks**:
- [ ] Add path interpolation before optimization to increase waypoint count
- [ ] Implement adaptive learning rate (line search or Armijo rule)
- [ ] Add momentum/velocity term for smoother convergence
- [ ] Improve convergence criteria (check relative cost improvement)
- [ ] Add early stopping if cost plateaus

**Files to modify**:
- `trajectory_optimizer.hh` - Update `optimize()` function
- `trajectory_optimizer_settings.hh` - Add adaptive rate parameters

**Success metric**: See >30% smoothness/acceleration improvements on test problems

---

#### 2.2 Better Time Parameterization
**Priority**: 🔴 Critical
**Effort**: Medium (3-4 days)
**Impact**: High - Enables time-optimal trajectories

**Tasks**:
- [ ] Implement velocity-bounded time allocation
- [ ] Add minimum-time objective to cost function
- [ ] Optimize timesteps between waypoints (not just positions)
- [ ] Add trapezoid velocity profile initialization
- [ ] Support maximum velocity/acceleration per joint

**New features**:
```cpp
// Time optimization objective
struct TimeOptimizationSettings {
    bool optimize_time = true;
    float weight_duration = 1.0F;
    bool use_velocity_bounds = true;
    bool use_trapezoid_profile = true;
};
```

**Files to modify**:
- `trajectory.hh` - Add time optimization methods
- `trajectory_cost.hh` - Add duration cost term
- `trajectory_optimizer.hh` - Optimize both positions and times

**Success metric**: Achieve 20-40% shorter trajectory durations while respecting velocity limits

---

#### 2.3 L-BFGS Quasi-Newton Optimizer
**Priority**: 🟡 High
**Effort**: Medium (4-5 days)
**Impact**: High - 5-10x faster convergence

**Tasks**:
- [ ] Implement L-BFGS with limited memory (m=10 typical)
- [ ] Store history of gradient differences and position updates
- [ ] Approximate Hessian inverse using two-loop recursion
- [ ] Add line search for step size selection
- [ ] Fallback to gradient descent if L-BFGS fails

**New file**: `src/impl/vamp/planning/trajectory_lbfgs.hh`

**Algorithm**:
```cpp
// L-BFGS two-loop recursion
1. Store last m gradient/position pairs (s_k, y_k)
2. Compute search direction using H_k approximation
3. Line search for optimal step size
4. Update trajectory and history buffers
```

**Files to create**:
- `trajectory_lbfgs.hh` - L-BFGS implementation
- `trajectory_line_search.hh` - Line search utilities

**Success metric**: Converge in 10-20 iterations vs 50-200 for gradient descent

---

### 🚀 Phase 3: SIMD Acceleration (MAXIMUM PERFORMANCE)

These changes unlock massive parallelization using VAMP's SIMD infrastructure.

#### 3.1 SIMD Batch Processing for Multi-Seed Optimization
**Priority**: 🟡 High
**Effort**: Large (7-10 days)
**Impact**: Very High - 4-8x throughput via parallel optimization

**Concept**:
```
Current: Optimize 1 trajectory at a time
Target:  Optimize 8 trajectories simultaneously (AVX) or 4 (NEON)
```

**Tasks**:
- [ ] Create `TrajectoryBatch<Robot, rake>` structure
- [ ] Store rake trajectories in SIMD-friendly layout
- [ ] Batch cost function evaluation (all rake trajectories at once)
- [ ] Batch gradient computation (vectorized finite differences)
- [ ] Batch constraint checking
- [ ] Multi-seed generation (from different planners + perturbations)
- [ ] Best trajectory selection after batch optimization

**New file**: `src/impl/vamp/planning/trajectory_batch.hh`

**Data structure**:
```cpp
template <typename Robot, std::size_t rake>
struct TrajectoryBatch {
    std::vector<typename Robot::template ConfigurationBlock<rake>> waypoints;
    std::vector<FloatVector<rake>> timestamps;

    // Evaluate costs for all rake trajectories in parallel
    auto compute_costs_batch(settings) -> FloatVector<rake>;

    // Compute gradients for all rake trajectories
    auto compute_gradients_batch(env, settings) -> TrajectoryBatch;
};
```

**Optimization flow**:
1. Seed from multiple VAMP planners (RRTC, PRM, FCIT)
2. Create rake variations per seed (small random perturbations)
3. Batch optimize all rake×seeds trajectories
4. Return best trajectory from entire population

**Files to create**:
- `trajectory_batch.hh` - Batch trajectory structure
- `trajectory_batch_cost.hh` - SIMD batch cost evaluation
- `trajectory_batch_optimizer.hh` - Batch optimization

**Success metric**: Optimize 32 trajectory seeds in <100ms total (vs 320ms sequential)

---

#### 3.2 Vectorized Gradient Computation
**Priority**: 🟢 Medium
**Effort**: Medium (3-4 days)
**Impact**: Medium - 4-8x faster gradient computation

**Current approach**:
```cpp
// Sequential: Perturb each dimension one at a time
for (j = 0; j < dimension; ++j) {
    perturb[j] += epsilon;
    grad[j] = (cost_perturbed - cost_base) / epsilon;
    perturb[j] -= epsilon;  // Reset
}
// Total: dimension × cost evaluations
```

**Vectorized approach**:
```cpp
// Parallel: Evaluate rake perturbations simultaneously
ConfigurationBlock<rake> perturbed_batch;  // rake configs at once
for (j = 0; j < dimension; j += rake) {
    // Perturb rake dimensions in parallel
    perturbed_batch = create_perturbed_block(base, j, epsilon);
    auto costs = eval_batch(perturbed_batch);  // SIMD evaluation
    extract_gradients(costs, j);
}
// Total: ceil(dimension / rake) × batch cost evaluations
```

**Tasks**:
- [ ] Implement batch cost evaluation for ConfigurationBlock
- [ ] Create perturbation block generator
- [ ] Extract gradients from SIMD cost vector
- [ ] Handle dimension not divisible by rake

**Files to modify**:
- `trajectory_optimizer.hh` - Vectorize `compute_gradient()`
- `trajectory_cost.hh` - Add batch cost evaluation

**Success metric**: 8x faster gradient computation on AVX (dimension=7 becomes 1 batch)

---

### 🎯 Phase 4: Robustness & Alternative Methods

Add alternative optimization methods that may work better than gradient descent.

#### 4.1 MPPI (Model Predictive Path Integral)
**Priority**: 🟢 Medium
**Effort**: Large (7-10 days)
**Impact**: High - Better local minima escape

**Algorithm**:
```
1. Generate population of K trajectory variations (noise perturbations)
2. Evaluate cost for each trajectory
3. Weight trajectories by exp(-cost/lambda) (Boltzmann)
4. Compute weighted combination for next iteration
5. Repeat until convergence
```

**Advantages**:
- Sampling-based (less prone to local minima than gradient descent)
- Naturally parallel (evaluate K trajectories independently)
- No gradient computation required
- Works well with non-differentiable costs

**Tasks**:
- [ ] Implement noise generation (Gaussian or correlated)
- [ ] Batch trajectory evaluation (K trajectories)
- [ ] Boltzmann weighting and trajectory combination
- [ ] Covariance adaptation
- [ ] Integration with SIMD batch processing

**New file**: `src/impl/vamp/planning/trajectory_mppi.hh`

**Parameters**:
```cpp
struct MPPISettings {
    std::size_t num_samples = 64;      // K trajectory samples
    float temperature = 1.0F;           // lambda in Boltzmann weight
    float noise_stddev = 0.1F;          // Perturbation magnitude
    std::size_t num_iterations = 50;
};
```

**Success metric**: Find better solutions on multi-modal cost landscapes

---

#### 4.2 Collision Avoidance via Distance Fields
**Priority**: 🟢 Medium
**Effort**: Large (7-10 days)
**Impact**: Medium - Smoother gradients near obstacles

**Current approach**: Binary collision check (in collision or not)
**Distance field approach**: Continuous distance to nearest obstacle

**Advantages**:
- Smooth gradients even near obstacles
- Attractive/repulsive forces for optimization
- Can penalize trajectories that are "too close" even if collision-free

**Tasks**:
- [ ] Compute signed distance field (SDF) for environment
- [ ] Efficient SDF queries for each waypoint
- [ ] Replace binary collision constraint with distance penalty
- [ ] Tune distance threshold and penalty weights

**New file**: `src/impl/vamp/planning/trajectory_sdf.hh`

**Cost modification**:
```cpp
// Instead of: collision ? 1000.0 : 0.0
// Use: smooth_penalty(distance, threshold)
float distance_cost = 0.0F;
for (waypoint : trajectory) {
    float d = sdf.query(waypoint);
    if (d < threshold) {
        distance_cost += (threshold - d)^2;  // Smooth penalty
    }
}
```

**Files to create**:
- `trajectory_sdf.hh` - Distance field utilities
- Modify `trajectory_constraints.hh` - Add distance-based constraints

**Success metric**: Smoother trajectories near obstacles, fewer optimizer failures

---

#### 4.3 STOMP (Stochastic Trajectory Optimization)
**Priority**: 🔵 Low
**Effort**: Large (7-10 days)
**Impact**: Medium - Alternative to MPPI

**Algorithm**:
Similar to MPPI but uses covariance matrix for noise generation.

**Tasks**:
- [ ] Implement covariance matrix estimation
- [ ] Generate correlated noise samples
- [ ] Cost-weighted trajectory updates
- [ ] Adaptive covariance updates

**New file**: `src/impl/vamp/planning/trajectory_stomp.hh`

**Decision point**: Evaluate MPPI performance first; only implement STOMP if needed

---

### 🎨 Phase 5: Usability & Integration

Make trajectory optimization easy to use and integrate with existing workflows.

#### 5.1 Unified Planner Interface
**Priority**: 🟡 High
**Effort**: Small (2-3 days)
**Impact**: High - Better user experience

**Goal**: Make trajectory optimizer a drop-in replacement for other planners

**Current usage**:
```cpp
// Step 1: Plan
auto rrtc_result = RRTC::solve(start, goal, env, settings, rng);

// Step 2: Optimize
auto opt_result = TrajectoryOptimizer::optimize_from_path(
    rrtc_result.path, env, opt_settings);
```

**Target usage**:
```cpp
// One-step planning with automatic trajectory optimization
auto result = solve<Panda, TrajectoryPlanner>(
    start, goal, env, settings, rng);
```

**Tasks**:
- [ ] Create `trajectory_planner.hh` wrapper
- [ ] Support seeding from multiple base planners
- [ ] Optional optimization (can be disabled)
- [ ] Unified settings structure

**New file**: `src/impl/vamp/planning/trajectory_planner.hh`

**Implementation**:
```cpp
template <typename Robot, std::size_t rake, std::size_t resolution>
struct TrajectoryPlanner {
    enum class BasePlanner { RRTC, PRM, FCIT, AORRTC };

    static auto solve(
        const Configuration& start,
        const Configuration& goal,
        const Environment& environment,
        const TrajectoryPlannerSettings& settings,
        RNG rng) -> PlanningResult<Robot>
    {
        // 1. Plan with base planner
        auto path = plan_with_base(base_planner, ...);

        // 2. Optionally optimize
        if (settings.optimize_trajectory) {
            return TrajectoryOptimizer::optimize_from_path(path, ...);
        }
        return path_to_result(path);
    }
};
```

**Success metric**: Single function call for planning + optimization

---

#### 5.2 Visualization & Debugging Tools
**Priority**: 🟢 Medium
**Effort**: Medium (3-4 days)
**Impact**: Medium - Better debugging and analysis

**Features**:
- [ ] Export trajectory to JSON/CSV
- [ ] Per-iteration cost history
- [ ] Constraint violation tracking
- [ ] Side-by-side comparison (seed vs optimized)
- [ ] Velocity/acceleration profiles plotting

**New file**: `src/impl/vamp/planning/trajectory_io.hh`

**Export formats**:
```json
{
  "waypoints": [[0.0, -0.785, ...], ...],
  "timestamps": [0.0, 0.1, 0.2, ...],
  "costs": {
    "smoothness": [10.5, 8.2, 5.1, ...],
    "acceleration": [15.3, 12.1, 8.7, ...],
    "path_length": [7.42, 7.35, 7.31, ...]
  },
  "iterations": 50
}
```

**Tasks**:
- [ ] JSON export for trajectories
- [ ] CSV export for plotting
- [ ] Python plotting utilities
- [ ] Real-time cost visualization

**Files to create**:
- `trajectory_io.hh` - Export utilities
- `scripts/python/plot_trajectory.py` - Plotting tool

**Success metric**: Easy visualization of optimization progress

---

#### 5.3 Robot-Specific Dynamics
**Priority**: 🔵 Low
**Effort**: Large (10+ days)
**Impact**: Medium - More realistic constraints

**Current**: Generic velocity/acceleration limits per joint
**Target**: Robot-specific dynamics models

**Features**:
- [ ] Load joint limits from robot specification
- [ ] Compute forward/inverse dynamics
- [ ] Torque limit constraints
- [ ] Dynamic feasibility checking
- [ ] Gravity compensation

**New file**: `src/impl/vamp/planning/trajectory_dynamics.hh`

**Requirements**:
- Robot mass/inertia properties
- Forward kinematics (already available)
- Inverse dynamics computation

**Per-robot implementation**:
```cpp
// Extend robot definitions (e.g., panda.hh)
struct Panda {
    // ... existing ...

    static constexpr std::array<float, dimension> max_velocities = {...};
    static constexpr std::array<float, dimension> max_accelerations = {...};
    static constexpr std::array<float, dimension> max_torques = {...};

    // Link masses and inertias
    static constexpr std::array<float, dimension> link_masses = {...};
};
```

**Success metric**: Dynamically feasible trajectories respecting torque limits

---

## Testing Strategy

### Unit Tests
- [ ] Trajectory time parameterization
- [ ] Cost function computation
- [ ] Constraint checking
- [ ] Gradient computation accuracy (finite difference validation)
- [ ] L-BFGS convergence
- [ ] SIMD batch operations

### Integration Tests
- [ ] End-to-end optimization on standard problems
- [ ] Multi-seed optimization
- [ ] Constraint satisfaction verification
- [ ] Performance benchmarks

### Benchmark Problems
1. **Cage problem** (current demo)
2. **Narrow passage** (tests constraint handling)
3. **Multi-obstacle** (tests collision avoidance)
4. **Long-range motion** (tests time optimization)

### Performance Targets
- Single trajectory optimization: <20ms
- 8-trajectory batch: <100ms
- 32-seed multi-start: <500ms
- Converge in <50 iterations (L-BFGS)

---

## Implementation Priorities

### 🔥 Immediate (Next 1-2 weeks)
Focus on making the current optimizer work well:
1. **Phase 2.1**: Fix gradient descent issues
2. **Phase 2.2**: Better time parameterization
3. **Phase 2.3**: L-BFGS optimizer

**Rationale**: These are foundational improvements with high impact and reasonable effort.

### 🚀 Short-term (Next 1-2 months)
Unlock SIMD parallelization:
1. **Phase 3.1**: SIMD batch processing
2. **Phase 3.2**: Vectorized gradients
3. **Phase 5.1**: Unified planner interface

**Rationale**: Maximum performance gains, makes trajectory optimization practical for real-time use.

### 🎯 Medium-term (3-6 months)
Add robustness and alternatives:
1. **Phase 4.1**: MPPI optimizer
2. **Phase 4.2**: Collision distance fields
3. **Phase 5.2**: Visualization tools

**Rationale**: Makes optimizer more robust and easier to use/debug.

### 🎨 Long-term (6+ months)
Polish and advanced features:
1. **Phase 4.3**: STOMP (if needed)
2. **Phase 5.3**: Robot dynamics
3. Additional features based on user feedback

---

## Success Metrics

### Performance
- [ ] <20ms single trajectory optimization
- [ ] <100ms batch optimization (8 trajectories)
- [ ] 5-10x speedup with L-BFGS vs gradient descent
- [ ] 30-50% smoothness/acceleration improvement

### Quality
- [ ] 100% collision-free optimized trajectories
- [ ] Velocity/acceleration limits satisfied
- [ ] 20-40% reduction in trajectory duration

### Usability
- [ ] One-line API for planning + optimization
- [ ] JSON/CSV export for analysis
- [ ] Python plotting utilities

### Robustness
- [ ] <5% optimization failures on benchmark suite
- [ ] Graceful fallback when optimization fails
- [ ] Deterministic results (given seed)

---

## Dependencies

### Internal (VAMP)
- ✅ SIMD vector infrastructure (`vector/`)
- ✅ Collision checking (`collision/`)
- ✅ Existing planners (`planning/rrtc.hh`, etc.)
- ✅ RNG utilities (`random/`)

### External (Already Available)
- ✅ Eigen3 (linear algebra)
- ✅ nigh (nearest neighbor)
- ✅ C++17 compiler with SIMD support

### Optional Future Dependencies
- [ ] JSON library (for export) - could use nlohmann/json
- [ ] Python bindings (for visualization) - existing nanobind setup

---

## Code Organization

### New Files Structure
```
src/impl/vamp/planning/
├── trajectory.hh                      ✅ (done)
├── trajectory_optimizer_settings.hh   ✅ (done)
├── trajectory_cost.hh                 ✅ (done)
├── trajectory_constraints.hh          ✅ (done)
├── trajectory_optimizer.hh            ✅ (done)
├── trajectory_lbfgs.hh               ⬜ (Phase 2.3)
├── trajectory_line_search.hh         ⬜ (Phase 2.3)
├── trajectory_batch.hh               ⬜ (Phase 3.1)
├── trajectory_batch_cost.hh          ⬜ (Phase 3.1)
├── trajectory_batch_optimizer.hh     ⬜ (Phase 3.1)
├── trajectory_mppi.hh                ⬜ (Phase 4.1)
├── trajectory_sdf.hh                 ⬜ (Phase 4.2)
├── trajectory_stomp.hh               ⬜ (Phase 4.3)
├── trajectory_planner.hh             ⬜ (Phase 5.1)
├── trajectory_io.hh                  ⬜ (Phase 5.2)
└── trajectory_dynamics.hh            ⬜ (Phase 5.3)

scripts/cpp/
├── trajectory_optimizer_example.cc   ✅ (done)
├── trajectory_batch_example.cc       ⬜ (Phase 3.1)
└── trajectory_mppi_example.cc        ⬜ (Phase 4.1)

scripts/python/
└── plot_trajectory.py                ⬜ (Phase 5.2)
```

---

## Quick Start Improvements (To Try Now)

Want to see better results immediately with current implementation?

### Modify the demo (`trajectory_optimizer_example.cc`):

```cpp
// After Step 2 (simplification), add:
simplify_result.path.interpolate_to_n_states(50);  // More waypoints for optimization

// In optimization settings:
opt_settings.max_iterations = 200;              // More iterations
opt_settings.learning_rate = 0.01F;             // Higher learning rate
opt_settings.weight_smoothness = 10.0F;         // Emphasize smoothness
opt_settings.weight_acceleration = 5.0F;        // Emphasize acceleration
opt_settings.gradient_epsilon = 1e-3F;          // Coarser finite differences
```

This will give actual smoothness improvements in the demo output.

---

## References & Inspiration

### Papers
- **CuRobo**: "CuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation" (Sundaralingam et al., 2023)
- **CHOMP**: "CHOMP: Covariant Hamiltonian Optimization for Motion Planning" (Ratliff et al., 2009)
- **STOMP**: "STOMP: Stochastic Trajectory Optimization for Motion Planning" (Kalakrishnan et al., 2011)
- **MPPI**: "Model Predictive Path Integral Control" (Williams et al., 2017)

### Existing Implementations
- CuRobo: https://github.com/NVlabs/curobo
- MoveIt STOMP: https://github.com/ros-planning/moveit
- OMPL trajectory optimization: https://ompl.kavrakilab.org/

---

## Questions & Decisions

### Open Questions
1. **Spline representation**: Should we add B-spline trajectory representation for analytical smoothness?
2. **GPU support**: Should we target CUDA/ROCm for even more parallelism?
3. **ROS integration**: Should we provide ROS2 message conversions?
4. **Multi-arm**: How to handle dual-arm robots?

### Design Decisions to Make
- [ ] Which optimizer to prioritize: L-BFGS or MPPI?
- [ ] Batch size: Fixed rake vs dynamic?
- [ ] Constraint handling: Penalty vs barrier vs augmented Lagrangian?
- [ ] Time representation: Fixed timesteps vs variable?

---

## Contributors & Acknowledgments

**Current Implementation**: Claude Code (AI Assistant) + User
**Based on**: VAMP library architecture
**Inspired by**: CuRobo trajectory optimization

---

## Appendix: Algorithm Pseudocode

### Current Gradient Descent Implementation
```python
def optimize(seed_trajectory, environment, settings):
    traj = seed_trajectory
    best_traj = traj
    best_cost = compute_cost(traj, settings)

    for iteration in range(settings.max_iterations):
        # Compute gradient via finite differences
        gradient = compute_gradient(traj, environment, settings)

        # Gradient descent update
        for i in range(1, len(traj.waypoints) - 1):  # Skip start/goal
            traj.waypoints[i] -= settings.learning_rate * gradient.waypoints[i]

        # Track best
        cost = compute_cost(traj, settings)
        if cost < best_cost:
            best_cost = cost
            best_traj = traj

        # Check convergence
        if max(abs(gradient)) < settings.convergence_threshold:
            break

    return best_traj
```

### Target L-BFGS Implementation
```python
def optimize_lbfgs(seed_trajectory, environment, settings, m=10):
    traj = seed_trajectory
    s_history = []  # Position differences
    y_history = []  # Gradient differences

    grad_prev = compute_gradient(traj, environment, settings)

    for iteration in range(settings.max_iterations):
        # L-BFGS two-loop recursion to compute search direction
        search_dir = two_loop_recursion(grad_prev, s_history, y_history)

        # Line search for step size
        alpha = line_search(traj, search_dir, environment, settings)

        # Update trajectory
        traj_new = traj + alpha * search_dir
        grad_new = compute_gradient(traj_new, environment, settings)

        # Update history (keep last m pairs)
        s_history.append(traj_new - traj)
        y_history.append(grad_new - grad_prev)
        if len(s_history) > m:
            s_history.pop(0)
            y_history.pop(0)

        traj = traj_new
        grad_prev = grad_new

        # Check convergence
        if norm(grad_new) < settings.convergence_threshold:
            break

    return traj
```

### Target SIMD Batch Implementation
```python
def optimize_batch(seed_paths, environment, settings, rake=8):
    # Create rake trajectories from seeds
    batch = TrajectoryBatch(rake)
    for i in range(rake):
        seed = seed_paths[i % len(seed_paths)]  # Cycle through seeds
        traj = Trajectory.from_path(seed)
        traj = add_random_perturbation(traj)  # Small variation
        batch.add_trajectory(traj, i)

    for iteration in range(settings.max_iterations):
        # Compute costs for all rake trajectories in parallel (SIMD)
        costs = batch.compute_costs_batch(environment, settings)

        # Compute gradients for all rake trajectories
        gradients = batch.compute_gradients_batch(environment, settings)

        # Update all rake trajectories
        batch.apply_gradients(gradients, settings.learning_rate)

        # Check convergence (any trajectory converged)
        if any(gradients.max_gradient < settings.convergence_threshold):
            break

    # Return best trajectory from batch
    best_idx = argmin(costs)
    return batch.get_trajectory(best_idx)
```

---

**Last Updated**: 2025-01-16
**Version**: 1.0
**Status**: Phase 1 Complete, Ready for Phase 2
