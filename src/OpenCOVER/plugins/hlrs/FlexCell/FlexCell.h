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
#include <cover/ui/CovconfigLink.h>

#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
// Mutex protects access to live positions shared between threads

struct RobotPosition {
    std::vector<double> positions;
    std::string timestampUtc;
    double timeOffset; // calculated from first timestamp
};

class FlexCell : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    FlexCell();
    ~FlexCell() override;

    bool update() override;
private:
    std::unique_ptr<opencover::dataclient::DummyClient> m_client;
    std::array<opencover::dataclient::ObserverHandle, 7> m_axisHandles;
    opencover::ui::EditFieldConfigValue *m_hostname = nullptr, *m_port = nullptr, *m_url = nullptr;
    opencover::ui::Menu *m_menu = nullptr;
    // Replay data
    std::vector<RobotPosition> m_recordedPositions;
    size_t m_currentPositionIndex = 0;
    double m_replayStartTime = -1.0;
    bool m_isReplaying = false;
    
    bool loadRecordedData(const std::string& filepath);
    void updateReplay(double currentTime);

    // SSE live streaming (preferred)
    void initSSE();
    void shutdownSSE();
    void sseConsumerLoop();
    static size_t sseWriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata);
    std::string m_endpointUrl;
    void* m_curl = nullptr;  // CURL* without including curl headers
    std::thread m_sseThread;
    std::atomic<bool> m_sseStop{false};
    std::atomic<bool> m_sseConnected{false};
    std::string m_sseBuffer;

    // Shared state for live positions
    mutable std::mutex m_rabbitMutex;
    std::deque<RobotPosition> m_livePositions;
    bool m_bend = false;
    int m_variant = -1;
    bool m_variantChanged = false;
    int m_bendAnimation = -1;
    bool m_partAttachedToRobot = true;
};

#endif // COVER_PLUGIN_FLEX_CELL_H