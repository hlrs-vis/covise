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

#include <rabbitmq-c/amqp.h>
#include <rabbitmq-c/tcp_socket.h>

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
    
    // Always use RabbitMQ when compiled with support
    std::cerr << "FlexCell: Initializing RabbitMQ live streaming..." << std::endl;
    initRabbitMQ();

}

FlexCell::~FlexCell()
{
    shutdownRabbitMQ();
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
            vrmlNode->send(i, static_cast<float>(currentPos.positions[i] / (2.0 * M_PI)));
        }
    }
}

bool FlexCell::update()
{
    if(flexCellNodes.empty())
        return false;
    
    double currentTime = cover->frameTime();
    
    // Priority: RabbitMQ live > replay > dummy client
    if (m_rabbitConnected)
    {
        std::lock_guard<std::mutex> lock(m_rabbitMutex);
        auto vrmlNode = *flexCellNodes.begin();
        if (!m_livePositions.empty())
        {
            const auto& pos = m_livePositions.back();  // Always use the latest
            
            for (size_t i = 0; i < std::min(pos.positions.size(), size_t(7)); ++i)
            {
                // Positions are in radians, normalize for full rotation
                vrmlNode->send(i, static_cast<float>(pos.positions[i] / (2.0 * M_PI)));
            }
            
            // No need to pop - consumer thread keeps buffer at size 1
        }
        if(m_bend)
        {
            vrmlNode->bend();
            m_bend = false;
        }
        if(m_variantChanged)
        {
            vrmlNode->switchWorkpiece(m_variant);
            m_variantChanged = false;
        }

    }
    else
    if (m_isReplaying)
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

void FlexCell::initRabbitMQ()
{
    // Read config from environment variables
    const char* host = std::getenv("RABBITMQ_HOST");
    const char* portStr = std::getenv("RABBITMQ_PORT");
    const char* user = std::getenv("RABBITMQ_USER");
    const char* pass = std::getenv("RABBITMQ_PASS");
    const char* vhost = std::getenv("RABBITMQ_VHOST");
    const char* queue = std::getenv("RABBITMQ_QUEUE");
    
    std::string hostStr = host ? host : "localhost";
    int port = portStr ? std::atoi(portStr) : 5672;
    std::string userStr = user ? user : "guest";
    std::string passStr = pass ? pass : "guest";
    std::string vhostStr = vhost ? vhost : "/";
    std::string queueStr = queue ? queue : "robot.positions";
    
    std::cerr << "FlexCell RabbitMQ config:\n"
              << "  Host:  " << hostStr << ":" << port << "\n"
              << "  VHost: " << vhostStr << "\n"
              << "  User:  " << userStr << "\n"
              << "  Queue: " << queueStr << std::endl;
    
    m_rabbitStop = false;
    m_rabbitThread = std::thread(&FlexCell::rabbitmqConsumerLoop, this);
}

void FlexCell::shutdownRabbitMQ()
{
    m_rabbitStop = true;
    if (m_rabbitThread.joinable())
    {
        m_rabbitThread.join();
    }
}

void FlexCell::rabbitmqConsumerLoop()
{
    // Read config
    const char* host = std::getenv("RABBITMQ_HOST");
    const char* portStr = std::getenv("RABBITMQ_PORT");
    const char* user = std::getenv("RABBITMQ_USER");
    const char* pass = std::getenv("RABBITMQ_PASS");
    const char* vhost = std::getenv("RABBITMQ_VHOST");
    const char* queue = std::getenv("RABBITMQ_QUEUE");
    
    std::string hostStr = host ? host : "localhost";
    int port = portStr ? std::atoi(portStr) : 5672;
    std::string userStr = user ? user : "guest";
    std::string passStr = pass ? pass : "guest";
    std::string vhostStr = vhost ? vhost : "/";
    std::string queueStr = queue ? queue : "robot.positions";
    
    // Connect to RabbitMQ
    m_rabbitConn = amqp_new_connection();
    amqp_socket_t* socket = amqp_tcp_socket_new(m_rabbitConn);
    if (!socket)
    {
        std::cerr << "FlexCell: Failed to create TCP socket" << std::endl;
        return;
    }
    
    int status = amqp_socket_open(socket, hostStr.c_str(), port);
    if (status != AMQP_STATUS_OK)
    {
        std::cerr << "FlexCell: Failed to open socket: " << amqp_error_string2(status) << std::endl;
        amqp_destroy_connection(m_rabbitConn);
        m_rabbitConn = nullptr;
        return;
    }
    
    amqp_rpc_reply_t reply = amqp_login(m_rabbitConn, vhostStr.c_str(), 0, 131072, 0,
                                         AMQP_SASL_METHOD_PLAIN,
                                         userStr.c_str(), passStr.c_str());
    if (reply.reply_type != AMQP_RESPONSE_NORMAL)
    {
        std::cerr << "FlexCell: RabbitMQ login failed" << std::endl;
        amqp_destroy_connection(m_rabbitConn);
        m_rabbitConn = nullptr;
        return;
    }
    
    amqp_channel_open(m_rabbitConn, 1);
    reply = amqp_get_rpc_reply(m_rabbitConn);
    if (reply.reply_type != AMQP_RESPONSE_NORMAL)
    {
        std::cerr << "FlexCell: Failed to open channel" << std::endl;
        amqp_connection_close(m_rabbitConn, AMQP_REPLY_SUCCESS);
        amqp_destroy_connection(m_rabbitConn);
        m_rabbitConn = nullptr;
        return;
    }
    
    // Declare queue (durable=1 to match server)
    amqp_queue_declare(m_rabbitConn, 1, amqp_cstring_bytes(queueStr.c_str()),
                       0, 1, 0, 0, amqp_empty_table);
    reply = amqp_get_rpc_reply(m_rabbitConn);
    if (reply.reply_type != AMQP_RESPONSE_NORMAL)
    {
        std::cerr << "FlexCell: Failed to declare queue" << std::endl;
        amqp_channel_close(m_rabbitConn, 1, AMQP_REPLY_SUCCESS);
        amqp_connection_close(m_rabbitConn, AMQP_REPLY_SUCCESS);
        amqp_destroy_connection(m_rabbitConn);
        m_rabbitConn = nullptr;
        return;
    }
    
    // Set QoS for minimal latency: prefetch=1 to process messages immediately
    amqp_basic_qos(m_rabbitConn, 1, 0, 1, 0);
    reply = amqp_get_rpc_reply(m_rabbitConn);
    if (reply.reply_type != AMQP_RESPONSE_NORMAL)
    {
        std::cerr << "FlexCell: Failed to set QoS" << std::endl;
    }
    
    // Start consuming
    std::string consumerTag = "flexcell-consumer";
    amqp_basic_consume(m_rabbitConn, 1, amqp_cstring_bytes(queueStr.c_str()),
                       amqp_cstring_bytes(consumerTag.c_str()),
                       0, 0, 0, amqp_empty_table);
    reply = amqp_get_rpc_reply(m_rabbitConn);
    if (reply.reply_type != AMQP_RESPONSE_NORMAL)
    {
        std::cerr << "FlexCell: Failed to start consumer" << std::endl;
        amqp_channel_close(m_rabbitConn, 1, AMQP_REPLY_SUCCESS);
        amqp_connection_close(m_rabbitConn, AMQP_REPLY_SUCCESS);
        amqp_destroy_connection(m_rabbitConn);
        m_rabbitConn = nullptr;
        return;
    }
    
    m_rabbitConnected = true;
    std::cerr << "FlexCell: Connected to RabbitMQ, consuming from '" << queueStr << "'" << std::endl;
    
    uint64_t messageCount = 0;
    
    // Consume loop
    while (!m_rabbitStop.load())
    {
        amqp_envelope_t envelope;
        amqp_maybe_release_buffers(m_rabbitConn);
        
        // Short timeout for low latency (100ms) but allow graceful shutdown
        struct timeval timeout = {0, 100000};  // 100ms timeout
        reply = amqp_consume_message(m_rabbitConn, &envelope, &timeout, 0);
        
        if (reply.reply_type == AMQP_RESPONSE_LIBRARY_EXCEPTION)
        {
            if (reply.library_error == AMQP_STATUS_TIMEOUT)
            {
                continue;  // Normal timeout
            }
            std::cerr << "FlexCell: Consumer error: " << amqp_error_string2(reply.library_error) << std::endl;
            break;
        }
        
        if (reply.reply_type != AMQP_RESPONSE_NORMAL)
        {
            continue;
        }
        
        ++messageCount;
        
        // Verify content type
        bool validContentType = false;
        if (envelope.message.properties._flags & AMQP_BASIC_CONTENT_TYPE_FLAG)
        {
            std::string contentType((char*)envelope.message.properties.content_type.bytes,
                                   envelope.message.properties.content_type.len);
            validContentType = (contentType == "application/json");
        }
        
        bool success = false;
        if (validContentType && envelope.message.body.len > 0)
        {
            try
            {
                std::string body((char*)envelope.message.body.bytes, envelope.message.body.len);
                json j = json::parse(body);
                
                RobotPosition pos;
                if(j.contains("bend"))
                {
                    m_bend = true;
                } 
                if(j.contains("variant") && j["variant"].is_number_integer())
                {
                    m_variant = j["variant"].get<int>();
                    m_variantChanged = true;
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
                
                // Add to live buffer - keep only latest for minimal latency
                {
                    std::lock_guard<std::mutex> lock(m_rabbitMutex);
                    
                    // Clear old positions and keep only the latest one
                    m_livePositions.clear();
                    m_livePositions.push_back(pos);
                }
                
                if (messageCount % 100 == 0)
                {
                    std::cerr << "FlexCell: Received " << messageCount << " messages" << std::endl;
                }
                
                success = true;
            }
            catch (const json::parse_error& e)
            {
                std::cerr << "FlexCell: JSON parse error: " << e.what() << std::endl;
            }
        }
        
        // Ack or Nack
        if (success)
        {
            amqp_basic_ack(m_rabbitConn, 1, envelope.delivery_tag, 0);
        }
        else
        {
            amqp_basic_nack(m_rabbitConn, 1, envelope.delivery_tag, 0, 0);
        }
        
        amqp_destroy_envelope(&envelope);
    }
    
    std::cerr << "FlexCell: Processed " << messageCount << " messages, shutting down..." << std::endl;
    
    // Cleanup
    amqp_basic_cancel(m_rabbitConn, 1, amqp_cstring_bytes(consumerTag.c_str()));
    amqp_channel_close(m_rabbitConn, 1, AMQP_REPLY_SUCCESS);
    amqp_connection_close(m_rabbitConn, AMQP_REPLY_SUCCESS);
    amqp_destroy_connection(m_rabbitConn);
    m_rabbitConn = nullptr;
    m_rabbitConnected = false;
}