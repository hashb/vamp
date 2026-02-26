#pragma once

#include <cstddef>

namespace vamp::planning
{
    struct TrajectoryOptimizerSettings
    {
        // Optimization parameters
        std::size_t max_iterations = 100;
        float learning_rate = 0.01F;
        float convergence_threshold = 1e-4F;

        // Objective weights (CuRobo-inspired)
        float weight_smoothness = 1.0F;      // Minimize jerk
        float weight_acceleration = 0.5F;    // Minimize acceleration
        float weight_path_length = 0.1F;     // Shorter geometric paths

        // Constraint penalties
        float collision_penalty = 1000.0F;
        float velocity_penalty = 100.0F;
        float acceleration_penalty = 100.0F;
        float joint_limit_penalty = 1000.0F;

        // Robot-specific scaling
        float velocity_scale = 1.0F;         // Max velocity scale
        float acceleration_scale = 1.0F;     // Max acceleration scale

        // Constraint enforcement
        bool enforce_collision_free = true;
        bool enforce_velocity_limits = true;
        bool enforce_acceleration_limits = true;
        bool enforce_joint_limits = true;

        // Initial time parameterization
        float dt_initial = 0.1F;             // Initial timestep for seeding
        bool use_distance_timing = true;     // Use distance-based timing

        // Gradient computation
        float gradient_epsilon = 1e-4F;      // Finite difference epsilon
    };
}  // namespace vamp::planning
