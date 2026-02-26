#pragma once

#include <vamp/planning/plan.hh>
#include <vamp/vector.hh>

namespace vamp::planning
{
    template <typename Robot>
    struct Trajectory
    {
        using Configuration = FloatVector<Robot::dimension>;
        static constexpr auto dimension = Robot::dimension;

        std::vector<Configuration> waypoints;
        std::vector<float> timestamps;  // Time at each waypoint

        [[nodiscard]] inline auto size() const noexcept -> std::size_t
        {
            return waypoints.size();
        }

        // Compute velocities via finite differences
        [[nodiscard]] inline auto velocities() const noexcept -> std::vector<Configuration>
        {
            std::vector<Configuration> vels;
            if (waypoints.size() < 2)
            {
                return vels;
            }

            vels.reserve(waypoints.size() - 1);
            for (auto i = 0U; i < waypoints.size() - 1; ++i)
            {
                const float dt = timestamps[i + 1] - timestamps[i];
                if (dt > 1e-6F)
                {
                    vels.emplace_back((waypoints[i + 1] - waypoints[i]) / dt);
                }
                else
                {
                    vels.emplace_back(Configuration{});
                }
            }
            return vels;
        }

        // Compute accelerations via finite differences of velocities
        [[nodiscard]] inline auto accelerations() const noexcept -> std::vector<Configuration>
        {
            auto vels = velocities();
            std::vector<Configuration> accels;
            if (vels.size() < 2)
            {
                return accels;
            }

            accels.reserve(vels.size() - 1);
            for (auto i = 0U; i < vels.size() - 1; ++i)
            {
                const float dt = timestamps[i + 1] - timestamps[i];
                if (dt > 1e-6F)
                {
                    accels.emplace_back((vels[i + 1] - vels[i]) / dt);
                }
                else
                {
                    accels.emplace_back(Configuration{});
                }
            }
            return accels;
        }

        // Compute jerks via finite differences of accelerations
        [[nodiscard]] inline auto jerks() const noexcept -> std::vector<Configuration>
        {
            auto accels = accelerations();
            std::vector<Configuration> jrks;
            if (accels.size() < 2)
            {
                return jrks;
            }

            jrks.reserve(accels.size() - 1);
            for (auto i = 0U; i < accels.size() - 1; ++i)
            {
                const float dt = timestamps[i + 1] - timestamps[i];
                if (dt > 1e-6F)
                {
                    jrks.emplace_back((accels[i + 1] - accels[i]) / dt);
                }
                else
                {
                    jrks.emplace_back(Configuration{});
                }
            }
            return jrks;
        }

        // Convert to Path (discarding time information)
        [[nodiscard]] inline auto to_path() const noexcept -> Path<Robot>
        {
            Path<Robot> path;
            path.reserve(waypoints.size());
            for (const auto &wp : waypoints)
            {
                path.emplace_back(wp);
            }
            return path;
        }

        // Create from Path with uniform time parameterization
        static inline auto from_path(const Path<Robot> &path, float dt = 0.1F) noexcept -> Trajectory<Robot>
        {
            Trajectory<Robot> traj;
            traj.waypoints.reserve(path.size());
            traj.timestamps.reserve(path.size());

            float t = 0.0F;
            for (const auto &config : path)
            {
                traj.waypoints.emplace_back(config);
                traj.timestamps.emplace_back(t);
                t += dt;
            }
            return traj;
        }

        // Create from Path with time scaled by distance
        static inline auto from_path_with_timing(
            const Path<Robot> &path,
            float velocity_scale = 1.0F) noexcept -> Trajectory<Robot>
        {
            Trajectory<Robot> traj;
            if (path.empty())
            {
                return traj;
            }

            traj.waypoints.reserve(path.size());
            traj.timestamps.reserve(path.size());

            float t = 0.0F;
            traj.waypoints.emplace_back(path[0]);
            traj.timestamps.emplace_back(t);

            for (auto i = 1U; i < path.size(); ++i)
            {
                const float distance = path[i - 1].distance(path[i]);
                const float dt = distance / velocity_scale;
                t += dt;
                traj.waypoints.emplace_back(path[i]);
                traj.timestamps.emplace_back(t);
            }

            return traj;
        }

        // Total trajectory duration
        [[nodiscard]] inline auto duration() const noexcept -> float
        {
            if (timestamps.empty())
            {
                return 0.0F;
            }
            return timestamps.back() - timestamps.front();
        }

        // Path length (geometric)
        [[nodiscard]] inline auto path_length() const noexcept -> float
        {
            if (waypoints.size() < 2)
            {
                return 0.0F;
            }

            float length = 0.0F;
            for (auto i = 0U; i < waypoints.size() - 1; ++i)
            {
                length += waypoints[i].distance(waypoints[i + 1]);
            }
            return length;
        }
    };
}  // namespace vamp::planning
