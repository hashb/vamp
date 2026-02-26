#pragma once

#include <chrono>
#include <vamp/collision/environment.hh>
#include <vamp/planning/plan.hh>
#include <vamp/planning/trajectory.hh>
#include <vamp/planning/trajectory_constraints.hh>
#include <vamp/planning/trajectory_cost.hh>
#include <vamp/planning/trajectory_optimizer_settings.hh>
#include <vamp/utils.hh>
#include <vamp/vector.hh>

namespace vamp::planning
{
    template <typename Robot, std::size_t rake, std::size_t resolution>
    struct TrajectoryOptimizer
    {
        using Configuration = typename Robot::Configuration;
        static constexpr auto dimension = Robot::dimension;

        // Compute gradient via finite differences
        static inline auto compute_gradient(
            const Trajectory<Robot> &traj,
            const collision::Environment<FloatVector<rake>> &environment,
            const TrajectoryOptimizerSettings &settings) noexcept -> Trajectory<Robot>
        {
            Trajectory<Robot> gradient;
            gradient.waypoints.reserve(traj.waypoints.size());
            gradient.timestamps = traj.timestamps;  // Same timing

            const float base_cost =
                TrajectoryCost<Robot>::compute_total_cost(traj, settings) +
                TrajectoryConstraints<Robot, rake, resolution>::compute_constraint_violation(
                    traj, environment, settings);

            // Compute gradient for each waypoint (except start/goal which are fixed)
            for (auto i = 0U; i < traj.waypoints.size(); ++i)
            {
                Configuration grad{};

                // Skip start and goal waypoints (they're fixed)
                if (i == 0 or i == traj.waypoints.size() - 1)
                {
                    gradient.waypoints.emplace_back(grad);
                    continue;
                }

                // Finite difference for each dimension
                Trajectory<Robot> traj_perturbed = traj;
                for (auto j = 0U; j < dimension; ++j)
                {
                    // Positive perturbation
                    auto temp_config = traj_perturbed.waypoints[i].to_array();
                    temp_config[j] += settings.gradient_epsilon;
                    traj_perturbed.waypoints[i] = Configuration(temp_config.data());

                    const float cost_plus =
                        TrajectoryCost<Robot>::compute_total_cost(traj_perturbed, settings) +
                        TrajectoryConstraints<Robot, rake, resolution>::compute_constraint_violation(
                            traj_perturbed, environment, settings);

                    // Compute gradient - store in array first
                    auto grad_arr = grad.to_array();
                    grad_arr[j] = (cost_plus - base_cost) / settings.gradient_epsilon;
                    grad = Configuration(grad_arr.data());

                    // Reset
                    traj_perturbed.waypoints[i] = traj.waypoints[i];
                }

                gradient.waypoints.emplace_back(grad);
            }

            return gradient;
        }

        // Optimize trajectory using gradient descent
        static inline auto optimize(
            const Trajectory<Robot> &seed_traj,
            const collision::Environment<FloatVector<rake>> &environment,
            const TrajectoryOptimizerSettings &settings) noexcept -> PlanningResult<Robot>
        {
            auto start_time = std::chrono::steady_clock::now();

            PlanningResult<Robot> result;
            Trajectory<Robot> current_traj = seed_traj;

            float best_cost = TrajectoryCost<Robot>::compute_total_cost(current_traj, settings);
            Trajectory<Robot> best_traj = current_traj;

            for (auto iter = 0U; iter < settings.max_iterations; ++iter)
            {
                result.iterations++;

                // Compute gradient
                auto gradient = compute_gradient(current_traj, environment, settings);

                // Gradient descent step
                Trajectory<Robot> next_traj = current_traj;
                float max_gradient = 0.0F;

                for (auto i = 1U; i < next_traj.waypoints.size() - 1; ++i)  // Skip start/goal
                {
                    auto waypoint_arr = next_traj.waypoints[i].to_array();
                    for (auto j = 0U; j < dimension; ++j)
                    {
                        const float grad = gradient.waypoints[i][{0, j}];
                        max_gradient = std::max(max_gradient, std::abs(grad));
                        waypoint_arr[j] -= settings.learning_rate * grad;
                    }
                    next_traj.waypoints[i] = Configuration(waypoint_arr.data());
                }

                // Compute new cost
                const float new_cost = TrajectoryCost<Robot>::compute_total_cost(next_traj, settings);

                // Accept if cost improved
                if (new_cost < best_cost)
                {
                    best_cost = new_cost;
                    best_traj = next_traj;
                    current_traj = next_traj;
                }
                else
                {
                    // Still update current for exploration, but track best
                    current_traj = next_traj;
                }

                // Check convergence
                if (max_gradient < settings.convergence_threshold)
                {
                    break;
                }
            }

            // Convert best trajectory to path
            result.path = best_traj.to_path();
            result.cost = best_traj.path_length();
            result.nanoseconds = vamp::utils::get_elapsed_nanoseconds(start_time);

            return result;
        }

        // Optimize from a seed path
        static inline auto optimize_from_path(
            const Path<Robot> &seed_path,
            const collision::Environment<FloatVector<rake>> &environment,
            const TrajectoryOptimizerSettings &settings) noexcept -> PlanningResult<Robot>
        {
            // Convert path to trajectory
            Trajectory<Robot> seed_traj;
            if (settings.use_distance_timing)
            {
                seed_traj = Trajectory<Robot>::from_path_with_timing(seed_path, settings.velocity_scale);
            }
            else
            {
                seed_traj = Trajectory<Robot>::from_path(seed_path, settings.dt_initial);
            }

            return optimize(seed_traj, environment, settings);
        }
    };
}  // namespace vamp::planning
