#pragma once

#include <vamp/planning/trajectory.hh>
#include <vamp/planning/trajectory_optimizer_settings.hh>
#include <vamp/vector.hh>

namespace vamp::planning
{
    template <typename Robot>
    struct TrajectoryCost
    {
        using Configuration = typename Robot::Configuration;
        static constexpr auto dimension = Robot::dimension;

        // Compute smoothness cost (sum of squared jerk norms)
        static inline auto compute_smoothness_cost(const Trajectory<Robot> &traj) noexcept -> float
        {
            const auto jrks = traj.jerks();
            if (jrks.empty())
            {
                return 0.0F;
            }

            float cost = 0.0F;
            for (const auto &jerk : jrks)
            {
                // Squared L2 norm of jerk
                for (auto i = 0U; i < dimension; ++i)
                {
                    const float ji = jerk[{0, i}];
                    cost += ji * ji;
                }
            }
            return cost;
        }

        // Compute acceleration cost (sum of squared acceleration norms)
        static inline auto compute_acceleration_cost(const Trajectory<Robot> &traj) noexcept -> float
        {
            const auto accels = traj.accelerations();
            if (accels.empty())
            {
                return 0.0F;
            }

            float cost = 0.0F;
            for (const auto &accel : accels)
            {
                for (auto i = 0U; i < dimension; ++i)
                {
                    const float ai = accel[{0, i}];
                    cost += ai * ai;
                }
            }
            return cost;
        }

        // Compute path length cost
        static inline auto compute_path_length_cost(const Trajectory<Robot> &traj) noexcept -> float
        {
            return traj.path_length();
        }

        // Combined objective function
        static inline auto compute_total_cost(
            const Trajectory<Robot> &traj,
            const TrajectoryOptimizerSettings &settings) noexcept -> float
        {
            float cost = 0.0F;

            if (settings.weight_smoothness > 0.0F)
            {
                cost += settings.weight_smoothness * compute_smoothness_cost(traj);
            }

            if (settings.weight_acceleration > 0.0F)
            {
                cost += settings.weight_acceleration * compute_acceleration_cost(traj);
            }

            if (settings.weight_path_length > 0.0F)
            {
                cost += settings.weight_path_length * compute_path_length_cost(traj);
            }

            return cost;
        }

        // Compute velocity cost (for debugging/analysis)
        static inline auto compute_velocity_cost(const Trajectory<Robot> &traj) noexcept -> float
        {
            const auto vels = traj.velocities();
            if (vels.empty())
            {
                return 0.0F;
            }

            float cost = 0.0F;
            for (const auto &vel : vels)
            {
                for (auto i = 0U; i < dimension; ++i)
                {
                    const float vi = vel[{0, i}];
                    cost += vi * vi;
                }
            }
            return cost;
        }
    };
}  // namespace vamp::planning
