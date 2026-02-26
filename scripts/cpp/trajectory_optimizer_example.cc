#include <array>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include <vamp/collision/factory.hh>
#include <vamp/planning/rrtc.hh>
#include <vamp/planning/simplify.hh>
#include <vamp/planning/trajectory.hh>
#include <vamp/planning/trajectory_cost.hh>
#include <vamp/planning/trajectory_optimizer.hh>
#include <vamp/planning/validate.hh>
#include <vamp/random/halton.hh>
#include <vamp/robots/panda.hh>

using Robot = vamp::robots::Panda;
static constexpr const std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;
using RRTC = vamp::planning::RRTC<Robot, rake, Robot::resolution>;
using TrajectoryOptimizer = vamp::planning::TrajectoryOptimizer<Robot, rake, Robot::resolution>;

// Start and goal configurations
static constexpr Robot::ConfigurationArray start = {0., -0.785, 0., -2.356, 0., 1.571, 0.785};
static constexpr Robot::ConfigurationArray goal = {2.35, 1., 0., -0.8, 0, 2.5, 0.785};

// Spheres for the cage problem
static const std::vector<std::array<float, 3>> problem = {
    {0.55, 0, 0.25},    {0.35, 0.35, 0.25},  {0, 0.55, 0.25},     {-0.55, 0, 0.25},
    {-0.35, -0.35, 0.25}, {0, -0.55, 0.25},    {0.35, -0.35, 0.25}, {0.35, 0.35, 0.8},
    {0, 0.55, 0.8},     {-0.35, 0.35, 0.8},  {-0.55, 0, 0.8},     {-0.35, -0.35, 0.8},
    {0, -0.55, 0.8},    {0.35, -0.35, 0.8},
};

static constexpr float radius = 0.2;

auto main(int, char **) -> int
{
    std::cout << "=== VAMP Trajectory Optimization Demo ===" << std::endl << std::endl;

    // Build sphere cage environment
    EnvironmentInput environment;
    for (const auto &sphere : problem)
    {
        environment.spheres.emplace_back(vamp::collision::factory::sphere::array(sphere, radius));
    }

    environment.sort();
    auto env_v = EnvironmentVector(environment);

    // Create RNG for planning
    auto rng = std::make_shared<vamp::rng::Halton<Robot>>();

    // Step 1: Plan with RRTC
    std::cout << "Step 1: Planning with RRTC..." << std::endl;
    vamp::planning::RRTCSettings rrtc_settings;
    rrtc_settings.range = 1.0;

    auto rrtc_result =
        RRTC::solve(Robot::Configuration(start), Robot::Configuration(goal), env_v, rrtc_settings, rng);

    if (rrtc_result.path.size() == 0)
    {
        std::cout << "RRTC planning failed!" << std::endl;
        return 1;
    }

    std::cout << "  RRTC path length: " << rrtc_result.path.cost() << std::endl;
    std::cout << "  RRTC path waypoints: " << rrtc_result.path.size() << std::endl;
    std::cout << "  RRTC time: " << rrtc_result.nanoseconds / 1e6 << " ms" << std::endl;
    std::cout << std::endl;

    // Step 2: Simplify path
    std::cout << "Step 2: Simplifying path..." << std::endl;
    vamp::planning::SimplifySettings simplify_settings;
    auto simplify_result =
        vamp::planning::simplify<Robot, rake, Robot::resolution>(rrtc_result.path, env_v, simplify_settings, rng);

    std::cout << "  Simplified path length: " << simplify_result.path.cost() << std::endl;
    std::cout << "  Simplified path waypoints: " << simplify_result.path.size() << std::endl;
    std::cout << "  Simplify time: " << simplify_result.nanoseconds / 1e6 << " ms" << std::endl;
    std::cout << std::endl;

    // Step 3: Convert to trajectory and compute initial costs
    std::cout << "Step 3: Converting to trajectory..." << std::endl;
    auto initial_traj = vamp::planning::Trajectory<Robot>::from_path_with_timing(simplify_result.path, 1.0F);

    std::cout << "  Initial trajectory duration: " << initial_traj.duration() << " s" << std::endl;
    std::cout << "  Initial trajectory waypoints: " << initial_traj.size() << std::endl;

    // Compute initial costs
    vamp::planning::TrajectoryOptimizerSettings opt_settings;
    float initial_smoothness =
        vamp::planning::TrajectoryCost<Robot>::compute_smoothness_cost(initial_traj);
    float initial_accel = vamp::planning::TrajectoryCost<Robot>::compute_acceleration_cost(initial_traj);
    float initial_length = vamp::planning::TrajectoryCost<Robot>::compute_path_length_cost(initial_traj);

    std::cout << "  Initial smoothness cost: " << initial_smoothness << std::endl;
    std::cout << "  Initial acceleration cost: " << initial_accel << std::endl;
    std::cout << "  Initial path length: " << initial_length << std::endl;
    std::cout << std::endl;

    // Step 4: Optimize trajectory
    std::cout << "Step 4: Optimizing trajectory..." << std::endl;
    opt_settings.max_iterations = 50;
    opt_settings.learning_rate = 0.001F;
    opt_settings.weight_smoothness = 1.0F;
    opt_settings.weight_acceleration = 0.5F;
    opt_settings.weight_path_length = 0.1F;
    opt_settings.gradient_epsilon = 1e-3F;

    auto optimized_result =
        TrajectoryOptimizer::optimize_from_path(simplify_result.path, env_v, opt_settings);

    std::cout << "  Optimization iterations: " << optimized_result.iterations << std::endl;
    std::cout << "  Optimization time: " << optimized_result.nanoseconds / 1e6 << " ms" << std::endl;
    std::cout << "  Optimized path length: " << optimized_result.path.cost() << std::endl;
    std::cout << "  Optimized path waypoints: " << optimized_result.path.size() << std::endl;
    std::cout << std::endl;

    // Compute final costs
    auto final_traj = vamp::planning::Trajectory<Robot>::from_path_with_timing(optimized_result.path, 1.0F);
    float final_smoothness = vamp::planning::TrajectoryCost<Robot>::compute_smoothness_cost(final_traj);
    float final_accel = vamp::planning::TrajectoryCost<Robot>::compute_acceleration_cost(final_traj);
    float final_length = vamp::planning::TrajectoryCost<Robot>::compute_path_length_cost(final_traj);

    std::cout << "Step 5: Results comparison:" << std::endl;
    std::cout << "  Smoothness: " << initial_smoothness << " -> " << final_smoothness
              << " (improvement: " << std::fixed << std::setprecision(1)
              << 100.0F * (initial_smoothness - final_smoothness) / initial_smoothness << "%)" << std::endl;
    std::cout << "  Acceleration: " << initial_accel << " -> " << final_accel
              << " (improvement: " << 100.0F * (initial_accel - final_accel) / initial_accel << "%)"
              << std::endl;
    std::cout << "  Path length: " << initial_length << " -> " << final_length
              << " (change: " << std::setprecision(2)
              << 100.0F * (final_length - initial_length) / initial_length << "%)" << std::endl;
    std::cout << std::endl;

    // Output final trajectory waypoints
    std::cout << "Final optimized trajectory waypoints:" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (const auto &config : optimized_result.path)
    {
        const auto &array = config.to_array();
        for (auto i = 0U; i < Robot::dimension; ++i)
        {
            std::cout << array[i];
            if (i < Robot::dimension - 1)
            {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;
    }

    return 0;
}
