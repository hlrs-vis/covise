#include "ConnectorTrafficRecording.h"

#include <algorithm>

#include <cover/coHud.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRMSController.h>

#include "Traffic.h"

inline double sumoAngleToMath(double angle)
{
    // See https://github.com/eclipse-sumo/sumo/issues/1372
    // Return a normal math angle that works with sin/cos, i.e. radians, counter-clockwise, from positive X-axis.
    return ((90.0 - angle) / 180.0) * M_PI;
}

ConnectorTrafficRecording::ConnectorTrafficRecording(const std::string &filename)
    : file_(filename, std::ios::binary)
{
    if (!file_)
        throw std::runtime_error("Unable to open FCD file");

    char magic[4];
    file_.read(magic, 4);

    if (!file_ || std::string(magic, 4) != "FCD1")
        throw std::runtime_error("Invalid FCD file");

    version_ = read<std::uint32_t>();
    vehicle_count_ = read<std::uint32_t>();
    timestep_count_ = read<std::uint32_t>();
    index_offset_ = read<std::uint64_t>();

    if (version_ != 1)
        throw std::runtime_error("Unsupported FCD version");

    ids_.reserve(vehicle_count_);

    for (std::uint32_t i = 0; i < vehicle_count_; ++i)
    {
        const auto length = read<std::uint16_t>();
        std::string id(length, '\0');
        file_.read(id.data(), length);

        if (!file_)
            throw std::runtime_error("Invalid vehicle dictionary");

        ids_.push_back(std::move(id));
    }

    file_.seekg(static_cast<std::streamoff>(index_offset_));

    offsets_.resize(timestep_count_);
    times_.resize(timestep_count_);

    for (std::uint32_t i = 0; i < timestep_count_; ++i)
        offsets_[i] = read<std::uint64_t>();

    // Read and cache each timestep's timestamp.
    for (std::uint32_t i = 0; i < timestep_count_; ++i)
    {
        file_.seekg(static_cast<std::streamoff>(offsets_[i]));
        times_[i] = read<double>();
    }

    opencover::coVRAnimationManager::instance()->setNumTimesteps(times_.size());
    opencover::coVRAnimationManager::instance()->setAnimationSpeed(5.0); // 0.2 per tick
    opencover::coVRAnimationManager::instance()->enableAnimation(true);
}

bool ConnectorTrafficRecording::isConnected() const
{
    return true;
}

bool ConnectorTrafficRecording::update(double deltaTime, double simulationDeltaTime)
{
    int frame = opencover::coVRAnimationManager::instance()->getAnimationFrame();
    if (times_[frame] == m_simulationTime)
    {
        return false;
    }

    m_simulationTime = times_[frame];
    m_updated = true;
    return true;
}

void ConnectorTrafficRecording::getSimulationState(SimulationState &state)
{
    if (m_updated)
    {
        auto it = std::lower_bound(times_.begin(), times_.end(), m_simulationTime);

        if (it == times_.end() || *it != m_simulationTime)
            return;

        const auto timestep = static_cast<std::size_t>(
            std::distance(times_.begin(), it));

        file_.seekg(static_cast<std::streamoff>(offsets_[timestep]));

        const auto time = read<double>();
        (void)time;

        const auto count = read<std::uint32_t>();
        vehicle_class_t vehicleClass("passenger");

        for (std::uint32_t i = 0; i < count; ++i)
        {
            const auto index = read<std::uint32_t>();

            if (index >= ids_.size())
                throw std::runtime_error("Invalid vehicle index");

            vehicle_id_t id(ids_[index]);

            float x = read<float>();
            float y = read<float>();
            float z = read<float>();
            float angle = read<float>();
            float speed = read<float>();
            state.vehicles[id] = VehicleState {
                id,
                vehicleClass,
                osg::Vec3d(x, y, z),
                sumoAngleToMath(angle),
                speed,
            };
        }

        m_updated = false;
    }
}
