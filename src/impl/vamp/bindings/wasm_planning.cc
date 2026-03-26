#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <emscripten/emscripten.h>

#include <vector>
#include <array>

#include <vamp/collision/factory.hh>
#include <vamp/planning/validate.hh>
#include <vamp/planning/rrtc.hh>
#include <vamp/planning/simplify.hh>
#include <vamp/robots/panda.hh>
#include <vamp/random/halton.hh>

extern "C" {

using Robot = vamp::robots::Panda;
static constexpr std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;
using RRTC = vamp::planning::RRTC<Robot, rake, Robot::resolution>;

// Start and goal configurations (Panda joint angles in radians)
static constexpr Robot::ConfigurationArray start = {0., -0.785, 0., -2.356, 0., 1.571, 0.785};
static constexpr Robot::ConfigurationArray goal = {2.35, 1., 0., -0.8, 0, 2.5, 0.785};

// Sphere cage obstacle positions (x, y, z)
static const std::vector<std::array<float, 3>> obstacle_positions = {
    {0.55, 0, 0.25},     {0.35, 0.35, 0.25},  {0, 0.55, 0.25},     {-0.55, 0, 0.25},
    {-0.35, -0.35, 0.25},{0, -0.55, 0.25},     {0.35, -0.35, 0.25}, {0.35, 0.35, 0.8},
    {0, 0.55, 0.8},      {-0.35, 0.35, 0.8},   {-0.55, 0, 0.8},     {-0.35, -0.35, 0.8},
    {0, -0.55, 0.8},     {0.35, -0.35, 0.8},
};

static constexpr float obstacle_radius = 0.2;

// Buffers for returning data to JavaScript
static std::vector<float> path_buffer;
static std::vector<float> sphere_buffer;
static float obstacle_buffer[14 * 4]; // 14 obstacles * (x, y, z, r)

EMSCRIPTEN_KEEPALIVE
int vamp_get_obstacle_count()
{
    return static_cast<int>(obstacle_positions.size());
}

EMSCRIPTEN_KEEPALIVE
float *vamp_get_obstacles()
{
    for (std::size_t i = 0; i < obstacle_positions.size(); ++i)
    {
        obstacle_buffer[i * 4 + 0] = obstacle_positions[i][0];
        obstacle_buffer[i * 4 + 1] = obstacle_positions[i][1];
        obstacle_buffer[i * 4 + 2] = obstacle_positions[i][2];
        obstacle_buffer[i * 4 + 3] = obstacle_radius;
    }
    return obstacle_buffer;
}

EMSCRIPTEN_KEEPALIVE
int vamp_get_robot_sphere_count()
{
    return static_cast<int>(Robot::n_spheres);
}

EMSCRIPTEN_KEEPALIVE
float *vamp_compute_fk(float j0, float j1, float j2, float j3, float j4, float j5, float j6)
{
    sphere_buffer.resize(Robot::n_spheres * 4);

    Robot::ConfigurationBlock<rake> q;
    // Set the first lane of each joint dimension
    q[0] = vamp::FloatVector<rake>(j0);
    q[1] = vamp::FloatVector<rake>(j1);
    q[2] = vamp::FloatVector<rake>(j2);
    q[3] = vamp::FloatVector<rake>(j3);
    q[4] = vamp::FloatVector<rake>(j4);
    q[5] = vamp::FloatVector<rake>(j5);
    q[6] = vamp::FloatVector<rake>(j6);

    Robot::Spheres<rake> spheres;
    Robot::sphere_fk(q, spheres);

    for (std::size_t i = 0; i < Robot::n_spheres; ++i)
    {
        sphere_buffer[i * 4 + 0] = spheres.x[i].element(0);
        sphere_buffer[i * 4 + 1] = spheres.y[i].element(0);
        sphere_buffer[i * 4 + 2] = spheres.z[i].element(0);
        sphere_buffer[i * 4 + 3] = spheres.r[i].element(0);
    }

    return sphere_buffer.data();
}

EMSCRIPTEN_KEEPALIVE
int vamp_plan()
{
    // Build sphere cage environment
    EnvironmentInput environment;
    for (const auto &sphere : obstacle_positions)
    {
        environment.spheres.emplace_back(vamp::collision::factory::sphere::array(sphere, obstacle_radius));
    }
    environment.sort();
    auto env_v = EnvironmentVector(environment);

    auto rng = std::make_shared<vamp::rng::Halton<Robot>>();

    vamp::planning::RRTCSettings rrtc_settings;
    rrtc_settings.range = 1.0;

    auto result =
        RRTC::solve(Robot::Configuration(start), Robot::Configuration(goal), env_v, rrtc_settings, rng);

    if (result.path.empty())
    {
        path_buffer.clear();
        return 0;
    }

    // Simplify path
    vamp::planning::SimplifySettings simplify_settings;
    auto simplify_result = vamp::planning::simplify<Robot, rake, Robot::resolution>(
        result.path, env_v, simplify_settings, rng);

    // Interpolate to smooth animation (target ~100 waypoints)
    simplify_result.path.interpolate_to_n_states(100);

    // Store path as flat array: path_len * 7 floats
    auto &path = simplify_result.path;
    path_buffer.resize(path.size() * Robot::dimension);
    for (std::size_t i = 0; i < path.size(); ++i)
    {
        const auto &arr = path[i].to_array();
        for (std::size_t j = 0; j < Robot::dimension; ++j)
        {
            path_buffer[i * Robot::dimension + j] = arr[j];
        }
    }

    return static_cast<int>(path.size());
}

EMSCRIPTEN_KEEPALIVE
float *vamp_get_path_data()
{
    return path_buffer.data();
}

EMSCRIPTEN_KEEPALIVE
int vamp_get_path_dof()
{
    return static_cast<int>(Robot::dimension);
}

// Legacy entry point for backward compatibility with wasm_planning.js
EMSCRIPTEN_KEEPALIVE
float vamp_wasm_planning()
{
    int path_len = vamp_plan();
    return static_cast<float>(path_len);
}

} // extern "C"
