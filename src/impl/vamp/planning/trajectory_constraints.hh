#pragma once

#include <vamp/collision/environment.hh>
#include <vamp/planning/trajectory.hh>
#include <vamp/planning/trajectory_optimizer_settings.hh>
#include <vamp/planning/validate.hh>
#include <vamp/vector.hh>

namespace vamp::planning
{
    template <typename Robot, std::size_t rake, std::size_t resolution>
    struct TrajectoryConstraints
    {
        using Configuration = typename Robot::Configuration;
        static constexpr auto dimension = Robot::dimension;

        // Check collision constraints for entire trajectory
        static inline auto check_collisions(
            const Trajectory<Robot> &traj,
            const collision::Environment<FloatVector<rake>> &environment) noexcept -> bool
        {
            // Check all waypoints and motions between them
            for (auto i = 0U; i < traj.waypoints.size(); ++i)
            {
                // Check waypoint collision by creating a block with the waypoint broadcasted
                typename Robot::template ConfigurationBlock<rake> block;
                for (auto j = 0U; j < dimension; ++j)
                {
                    block[j] = traj.waypoints[i].broadcast(j);
                }

                const bool valid = (environment.attachments) ?
                    Robot::template fkcc_attach<rake>(environment, block) :
                    Robot::template fkcc<rake>(environment, block);

                if (not valid)
                {
                    return false;
                }

                // Check motion between waypoints
                if (i < traj.waypoints.size() - 1)
                {
                    if (not validate_motion<Robot, rake, resolution>(
                            traj.waypoints[i], traj.waypoints[i + 1], environment))
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        // Check velocity limits (per joint)
        static inline auto check_velocity_limits(
            const Trajectory<Robot> &traj,
            float velocity_scale) noexcept -> bool
        {
            const auto vels = traj.velocities();
            if (vels.empty())
            {
                return true;
            }

            // Simple check: velocity magnitude per joint
            for (const auto &vel : vels)
            {
                for (auto i = 0U; i < dimension; ++i)
                {
                    if (std::abs(vel[{0, i}]) > velocity_scale)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        // Check acceleration limits (per joint)
        static inline auto check_acceleration_limits(
            const Trajectory<Robot> &traj,
            float acceleration_scale) noexcept -> bool
        {
            const auto accels = traj.accelerations();
            if (accels.empty())
            {
                return true;
            }

            for (const auto &accel : accels)
            {
                for (auto i = 0U; i < dimension; ++i)
                {
                    if (std::abs(accel[{0, i}]) > acceleration_scale)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        // Check joint limits using robot-specific bounds
        static inline auto check_joint_limits(const Trajectory<Robot> &traj) noexcept -> bool
        {
            // Use Robot's joint limit arrays (e.g., Panda::s_a, Panda::s_m)
            for (const auto &waypoint : traj.waypoints)
            {
                for (auto i = 0U; i < dimension; ++i)
                {
                    const float joint_val = waypoint[{0, i}];
                    const float lower = Robot::s_a[i];
                    const float upper = Robot::s_m[i];

                    if (joint_val < lower or joint_val > upper)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        // Compute total constraint violation (for penalty-based optimization)
        static inline auto compute_constraint_violation(
            const Trajectory<Robot> &traj,
            const collision::Environment<FloatVector<rake>> &environment,
            const TrajectoryOptimizerSettings &settings) noexcept -> float
        {
            float violation = 0.0F;

            // Collision violation
            if (settings.enforce_collision_free)
            {
                if (not check_collisions(traj, environment))
                {
                    violation += settings.collision_penalty;
                }
            }

            // Velocity violation
            if (settings.enforce_velocity_limits)
            {
                if (not check_velocity_limits(traj, settings.velocity_scale))
                {
                    violation += settings.velocity_penalty;
                }
            }

            // Acceleration violation
            if (settings.enforce_acceleration_limits)
            {
                if (not check_acceleration_limits(traj, settings.acceleration_scale))
                {
                    violation += settings.acceleration_penalty;
                }
            }

            // Joint limit violation
            if (settings.enforce_joint_limits)
            {
                if (not check_joint_limits(traj))
                {
                    violation += settings.joint_limit_penalty;
                }
            }

            return violation;
        }

        // Check all constraints
        static inline auto check_all_constraints(
            const Trajectory<Robot> &traj,
            const collision::Environment<FloatVector<rake>> &environment,
            const TrajectoryOptimizerSettings &settings) noexcept -> bool
        {
            if (settings.enforce_collision_free and not check_collisions(traj, environment))
            {
                return false;
            }

            if (settings.enforce_velocity_limits and
                not check_velocity_limits(traj, settings.velocity_scale))
            {
                return false;
            }

            if (settings.enforce_acceleration_limits and
                not check_acceleration_limits(traj, settings.acceleration_scale))
            {
                return false;
            }

            if (settings.enforce_joint_limits and not check_joint_limits(traj))
            {
                return false;
            }

            return true;
        }
    };
}  // namespace vamp::planning
