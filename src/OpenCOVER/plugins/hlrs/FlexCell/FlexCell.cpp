#include "FlexCell.h"
#include "VrmlNodes.h"
#include <vrml97/vrml/VrmlNamespace.h>
#include <array>
#include <fstream>
#include <nlohmann/json.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <util/threadname.h>

#include <curl/curl.h>

using namespace opencover;
using json = nlohmann::json;

constexpr std::array<const char*, 7> axisNames = {"Achse1", "Achse2", "Achse3", "Achse4", "Achse5", "Achse6", "Achse7"};

COVERPLUGIN(FlexCell)

FlexCell::FlexCell()
: opencover::coVRPlugin(COVER_PLUGIN_NAME)
{
    vrml::VrmlNamespace::addBuiltIn(vrml::VrmlNode::defineType<FlexCellNode>());
    
    m_client = std::make_unique<opencover::dataclient::DummyClient>("FlexCell");
    for (size_t i = 0; i < axisNames.size(); ++i)
    {
        m_axisHandles[i] = m_client->observeNode(axisNames[i]);
    }
    m_client->connect();
    
    // Initialize SSE live streaming
    std::cerr << "FlexCell: Initializing SSE live streaming..." << std::endl;
    initSSE();
}

FlexCell::~FlexCell()
{
    shutdownSSE();
}

bool FlexCell::loadRecordedData(const std::string& filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }
    
    json root;
    try
    {
        file >> root;
    }
    catch (const json::exception& e)
    {
        std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
        return false;
    }
    
    if (!root.contains("robot_positions") || !root["robot_positions"].is_array())
    {
        std::cerr << "Invalid JSON structure" << std::endl;
        return false;
    }
    
    const auto& positions = root["robot_positions"];
    m_recordedPositions.reserve(positions.size());
    
    // Parse first timestamp to calculate offsets
    std::string firstTimestamp = positions[0]["timestampUtc"].get<std::string>();
    auto parseTimestamp = [](const std::string& ts) -> double {
        // Parse ISO 8601 format: 2025-10-30T14:12:19.852725Z
        std::tm tm = {};
        std::istringstream ss(ts);
        ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
        double seconds = static_cast<double>(mktime(&tm));
        
        // Parse fractional seconds
        size_t dotPos = ts.find('.');
        if (dotPos != std::string::npos && ts.size() > dotPos + 1)
        {
            std::string fraction = ts.substr(dotPos + 1);
            fraction = fraction.substr(0, fraction.find('Z'));
            seconds += std::stod("0." + fraction);
        }
        return seconds;
    };
    
    double firstTime = parseTimestamp(firstTimestamp);
    
    for (const auto& pos : positions)
    {
        RobotPosition robotPos;
        robotPos.timestampUtc = pos["timestampUtc"].get<std::string>();
        robotPos.timeOffset = parseTimestamp(robotPos.timestampUtc) - firstTime;
        
        if (pos.contains("positions") && pos["positions"].is_array())
        {
            for (const auto& p : pos["positions"])
            {
                robotPos.positions.push_back(p.get<double>());
            }
        }
        
        m_recordedPositions.push_back(robotPos);
    }
    
    return true;
}

void FlexCell::updateReplay(double currentTime)
{
    if (!m_isReplaying || m_recordedPositions.empty())
        return;
    
    // Initialize replay start time
    if (m_replayStartTime < 0)
    {
        m_replayStartTime = currentTime;
        m_currentPositionIndex = 0;
    }
    
    double elapsedTime = currentTime - m_replayStartTime;
    
    // Find the appropriate position for current time
    while (m_currentPositionIndex < m_recordedPositions.size() - 1 &&
           m_recordedPositions[m_currentPositionIndex + 1].timeOffset <= elapsedTime)
    {
        m_currentPositionIndex++;
    }
    
    // Loop replay
    if (m_currentPositionIndex >= m_recordedPositions.size() - 1)
    {
        m_replayStartTime = currentTime;
        m_currentPositionIndex = 0;
    }
    
    // Send current positions to VRML nodes
    if (!flexCellNodes.empty())
    {
        auto vrmlNode = *flexCellNodes.begin();
        const auto& currentPos = m_recordedPositions[m_currentPositionIndex];
        
        for (size_t i = 0; i < std::min(currentPos.positions.size(), size_t(7)); ++i)
        {
            // Convert radians to normalized value (0-1) for full rotation
            auto norm = currentPos.positions[i] / (2.0 * M_PI);
            if(norm < 0.0) norm += 1.0;


            vrmlNode->send(i, static_cast<float>(norm));
            std::cerr << "axis " << i << " pos: " << norm << std::endl; 
        }
    }
}

bool FlexCell::update()
{
    if(flexCellNodes.empty())
        return false;
    
    double currentTime = cover->frameTime();
    
    // Priority: SSE live > replay > dummy client
    if (m_sseConnected)
    {
        std::lock_guard<std::mutex> lock(m_rabbitMutex);
        auto vrmlNode = *flexCellNodes.begin();
        if (!m_livePositions.empty())
        {
            const auto& pos = m_livePositions.back();  // Always use the latest
            
            for (size_t i = 0; i < std::min(pos.positions.size(), size_t(7)); ++i)
            {
                // Positions are in radians, normalize for full rotation
                auto norm = pos.positions[i] / (2.0 * M_PI);
                if(norm < 0.0) norm += 1.0;
                vrmlNode->send(i, static_cast<float>(norm));
            }
        }
        if(m_bend)
        {
            vrmlNode->bend();
            m_bend = false;
        }
        if(m_variant != -1)
        {
            vrmlNode->switchWorkpiece(m_variant);
            m_variantChanged = false;
        }
        if(m_bendAnimation > 0)
        {
            vrmlNode->bendAnimation(m_bendAnimation);
            m_bendAnimation = -1;
        }
        if(m_partAttachedToRobot > 0)
        {
            m_partAttachedToRobot ? vrmlNode->attachPartToRobot(m_variant) : vrmlNode->detachPartToRobot(m_variant);
            m_partAttachedToRobot = -1;
        }
    }
    else if (m_isReplaying)
    {
        updateReplay(currentTime);
    }
    else if (m_client && m_client->isConnected())
    {
        // Live mode - original code
        auto vrmlNode = *flexCellNodes.begin();
        for (size_t i = 0; i < axisNames.size(); ++i)
        {
            double value = m_client->getNumericScalar(m_axisHandles[i]);
            vrmlNode->send(i, static_cast<float>(value / 360));
        }
    }

    return true;
}

void FlexCell::initSSE()
{
    // Read SSE endpoint from environment variable
    const char* endpoint = std::getenv("SSE_ENDPOINT");
    std::string endpointStr = endpoint ? endpoint : "http://localhost:8000/events";
    
    std::cerr << "FlexCell SSE config:\n"
              << "  Endpoint: " << endpointStr << std::endl;
    
    curl_global_init(CURL_GLOBAL_ALL);
    m_sseStop = false;
    m_sseThread = std::thread(&FlexCell::sseConsumerLoop, this);
}

void FlexCell::shutdownSSE()
{
    m_sseStop = true;
    if (m_sseThread.joinable())
    {
        m_sseThread.join();
    }
    curl_global_cleanup();
}

size_t FlexCell::sseWriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata)
{
    FlexCell* self = static_cast<FlexCell*>(userdata);
    size_t totalSize = size * nmemb;
    self->m_sseConnected = true;
    self->m_sseBuffer.append(ptr, totalSize);
    
    // Process complete SSE messages (lines ending with \n\n)
    size_t pos = 0;
    while ((pos = self->m_sseBuffer.find("\n\n")) != std::string::npos)
    {
        std::string message = self->m_sseBuffer.substr(0, pos);
        self->m_sseBuffer.erase(0, pos + 2);
        
        // Parse SSE message format
        std::istringstream stream(message);
        std::string line;
        std::string eventType;
        std::string data;
        
        while (std::getline(stream, line))
        {
            // Remove carriage return if present
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            
            if (line.substr(0, 6) == "event:")
            {
                eventType = line.substr(6);
                // Trim whitespace
                eventType.erase(0, eventType.find_first_not_of(" \t"));
            }
            else if (line.substr(0, 5) == "data:")
            {
                data = line.substr(5);
                // Trim whitespace
                data.erase(0, data.find_first_not_of(" \t"));
            }
        }
        
        // Process the data
        if (!data.empty())
        {
            try
            {
                json j = json::parse(data);
                
                RobotPosition pos;
                
                if (j.contains("partInBendTool") && j["partInBendTool"].is_boolean() && j["partInBendTool"].get<bool>())
                {
                    self->m_bend = true;
                }

                if (j.contains("partState") && j["partState"].is_number_integer())
                {
                    self->m_variant = j["partState"].get<int>();
                    self->m_variantChanged = true;
                }
                if (j.contains("bendAnimation") && j["bendAnimation"].is_number_integer())
                {
                    self->m_bendAnimation = j["bendAnimation"].get<int>();
                }
                if (j.contains("partAttachedToRobot") && j["partAttachedToRobot"].is_boolean())
                {
                    self->m_partAttachedToRobot = j["partAttachedToRobot"].get<bool>();
                }
                
                if (j.contains("timestampUtc") && j["timestampUtc"].is_string())
                {
                    pos.timestampUtc = j["timestampUtc"].get<std::string>();
                }
                
                if (j.contains("positions") && j["positions"].is_array())
                {
                    for (const auto& p : j["positions"])
                    {
                        if (p.is_number())
                        {
                            pos.positions.push_back(p.get<double>());
                        }
                    }
                }
                
                // Update live buffer - keep only latest for minimal latency
                {
                    std::lock_guard<std::mutex> lock(self->m_rabbitMutex);
                    self->m_livePositions.clear();
                    self->m_livePositions.push_back(pos);
                }
            }
            catch (const json::parse_error& e)
            {
                std::cerr << "FlexCell: JSON parse error: " << e.what() << std::endl;
            }
        }
    }
    
    return totalSize;
}   

void FlexCell::sseConsumerLoop()
{
    covise::setThreadName("sse_listener");
    const char* endpoint = std::getenv("SSE_ENDPOINT");
    std::string endpointStr = endpoint ? endpoint : "http://localhost:8000/events";
    
    m_curl = curl_easy_init();
    if (!m_curl)
    {
        std::cerr << "FlexCell: Failed to initialize CURL" << std::endl;
        return;
    }
    
    CURL* curl = static_cast<CURL*>(m_curl);
    
    // Configure CURL for SSE
    curl_easy_setopt(curl, CURLOPT_URL, endpointStr.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, sseWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L); // No overall timeout for normal streaming
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

    // Debugging + robustness options
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);                  // verbose logs to stderr
    curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4); // force IPv4 if IPv6/route issues
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);            // enable TCP keepalive
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);          // treat almost-idle connection as slow
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 30L);          // ...if below limit for 30s -> abort
    
    // Set headers for SSE
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Accept: text/event-stream");
    headers = curl_slist_append(headers, "Cache-Control: no-cache");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    std::cerr << "FlexCell: Connecting to SSE endpoint: " << endpointStr << std::endl;
    
    while (!m_sseStop.load())
    {
        m_sseBuffer.clear();
        CURLcode res = curl_easy_perform(curl);
        
        if (res == CURLE_OK)
        {
            m_sseConnected = true;
            std::cerr << "FlexCell: SSE stream ended normally" << std::endl;
        }
        else
        {
            m_sseConnected = false;
            std::cerr << "FlexCell: SSE connection error: " << curl_easy_strerror(res) << std::endl;
        }
        
        // Reconnect after delay if not stopping
        if (!m_sseStop.load())
        {
            std::cerr << "FlexCell: Reconnecting to SSE in 5 seconds..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    m_curl = nullptr;
    m_sseConnected = false;
    
    std::cerr << "FlexCell: SSE consumer thread stopped" << std::endl;
}