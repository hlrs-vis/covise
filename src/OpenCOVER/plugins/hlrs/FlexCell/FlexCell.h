#ifndef COVER_PLUGIN_FLEX_CELL_H
#define COVER_PLUGIN_FLEX_CELL_H

#include <cover/coVRPluginSupport.h>
#include <DataClient/DummyClient.h>
#include <memory>
#include <vector>
#include <string>
#include <deque>
#include <thread>
#include <atomic>
#include <mutex>

// Forward declarations for RabbitMQ types
typedef struct amqp_connection_state_t_ *amqp_connection_state_t;

struct RobotPosition {
    std::vector<double> positions;
    std::string timestampUtc;
    double timeOffset; // calculated from first timestamp
};

class FlexCell : public opencover::coVRPlugin
{
public:
    FlexCell();
    ~FlexCell() override;

    bool update() override;
private:
    std::unique_ptr<opencover::dataclient::DummyClient> m_client;
    std::array<opencover::dataclient::ObserverHandle, 7> m_axisHandles;
    
    // Replay data
    std::vector<RobotPosition> m_recordedPositions;
    size_t m_currentPositionIndex = 0;
    double m_replayStartTime = -1.0;
    bool m_isReplaying = false;
    
    bool loadRecordedData(const std::string& filepath);
    void updateReplay(double currentTime);

    // RabbitMQ live streaming
    void initRabbitMQ();
    void shutdownRabbitMQ();
    void rabbitmqConsumerLoop();
    
    amqp_connection_state_t m_rabbitConn = nullptr;
    std::thread m_rabbitThread;
    std::atomic<bool> m_rabbitStop{false};
    
    mutable std::mutex m_rabbitMutex;
    std::deque<RobotPosition> m_livePositions;
    bool m_bend = false;
    int m_variant = 0;
    bool m_rabbitConnected = false;
    bool m_variantChanged = false;
};

#endif // COVER_PLUGIN_FLEX_CELL_H