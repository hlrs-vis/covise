#include "Drehgeber.h"

#include <cover/coVRPluginSupport.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlNodeTransform.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/System.h>

#include <util/coExport.h>
#include <iostream>
#include <vector>
using namespace vrml;


class VrmlNodeDrehgeber;
std::vector<VrmlNodeDrehgeber*> drehgebers;

class PLUGINEXPORT VrmlNodeDrehgeber : public vrml::VrmlNodeChild
{
public:
    static void initFields(VrmlNodeDrehgeber *node, vrml::VrmlNodeType *t)
    {
        VrmlNodeChild::initFields(node, t);
        if(t)
            t->addEventOut("angle", VrmlField::SFFLOAT);
    }
    
    static const char* typeName() { return "Drehgeber"; }

    VrmlNodeDrehgeber(VrmlScene* scene) : VrmlNodeChild(scene, typeName()), m_index(drehgebers.size())
    {

        std::cerr << "vrml Machine node created" << std::endl;
        drehgebers.push_back(this);
    }

    ~VrmlNodeDrehgeber()
    {
        drehgebers.erase(drehgebers.begin() + m_index);
    }

    void setAngle(VrmlSFFloat position)
    {
        auto t = System::the->time();
        eventOut(t, "angle", position);
    }

private:
    size_t m_index = 0;
};


static std::string trim(const std::string &s)
{
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])))
        ++b;

    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
        --e;

    return s.substr(b, e - b);
}

static int parseComNumber(const std::string &deviceId)
{
    // Expects "COM12" -> 12
    if (deviceId.size() < 4 || deviceId.rfind("COM", 0) != 0)
        return -1;

    try
    {
        return std::stoi(deviceId.substr(3));
    }
    catch (...)
    {
        return -1;
    }
}

std::map<int, std::string> getComPortDescriptions()
{
    std::map<int, std::string> result;
#ifdef _WIN32
    // Output format per line: COMx<TAB>Description
    const char *cmd =
        "powershell -NoProfile -Command "
        "\"Get-PnpDevice -Class Ports -PresentOnly -ErrorAction SilentlyContinue | "
        "ForEach-Object { "
        "$name = if ($_.FriendlyName) { $_.FriendlyName } elseif ($_.Name) { $_.Name } else { '' }; "
        "if ($name -match '\\((COM\\d+)\\)') { "
        "[string]::Format('{0}{1}{2}', $matches[1], [char]9, $name) "
        "} "
        "}\"";



        
    std::unique_ptr<FILE, int (*)(FILE *)> pipe(_popen(cmd, "r"), _pclose);
    if (!pipe)
        return result;

    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), pipe.get()))
    {
        std::string line = trim(buffer);
        if (line.empty())
        {
            std::cerr << "[Drehgeber] Serial port parse: ignoring empty line" << std::endl;
            continue;
        }

        size_t tabPos = line.find('\t');
        std::string deviceId;
        std::string description;

        if (tabPos != std::string::npos)
        {
            deviceId = trim(line.substr(0, tabPos));      // COM12
            description = trim(line.substr(tabPos + 1));  // USB Serial Device ...
        }
        else
        {
            // Fallback for unexpected output format, e.g. only a friendly name like "USB ... (COM12)"
            size_t comStart = line.rfind("(COM");
            size_t comEnd = (comStart != std::string::npos) ? line.find(')', comStart) : std::string::npos;
            if (comStart != std::string::npos && comEnd != std::string::npos && comEnd > comStart + 1)
            {
                deviceId = line.substr(comStart + 1, comEnd - comStart - 1); // COM12
                description = line;
                std::cerr << "[Drehgeber] Serial port parse: no tab found, used fallback for line: " << line << std::endl;
            }
            else
            {
                std::cerr << "[Drehgeber] Serial port parse: could not parse line: " << line << std::endl;
                continue;
            }
        }

        int comNumber = parseComNumber(deviceId);
        if (comNumber >= 0 && !description.empty())
        {
            result[comNumber] = description;
            std::cerr << "[Drehgeber] Serial port parse: COM" << comNumber << " -> " << description << std::endl;
        }
        else
        {
            std::cerr << "[Drehgeber] Serial port parse: invalid entry: deviceId='" << deviceId
                      << "', description='" << description << "'" << std::endl;
        }
    }
#endif
    return result;
}

COVERPLUGIN(Drehgeber)

Drehgeber::Drehgeber()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("Drehgeber_owner", cover->ui)
, m_config(config())
, m_menu(new ui::Menu("Drehgeber", this))
, m_rotator(new ui::Slider(m_menu, "rotation_angle_in_degree")) 
, m_serialDeviceHint(std::make_unique<ui::EditFieldConfigValue>(m_menu, "serialDeviceHint", "", *m_config, "Serial")) 
, m_baudrate(std::make_unique<ui::EditFieldConfigValue>(m_menu, "baudrate", "115200", *m_config, "Serial")) 
, m_delay(std::make_unique<ui::SliderConfigValue>(m_menu, "delay", 0, *m_config, "Serial")) 
{
    std::string serialDevice = "\\\\.\\COM12";
    auto ports = getComPortDescriptions();
    auto hint = m_serialDeviceHint->getValue();
    if(!hint.empty())
    {
        // Try to find a port matching the hint
        for (const auto& [comNumber, description] : ports)
        {
            if (description.find(hint) != std::string::npos)
            {
                serialDevice = std::string("\\\\.\\COM") + std::to_string(comNumber);
                break;
            }
        }
    }
    m_serialDevice = std::make_unique<ui::EditFieldConfigValue>(m_menu, "serialDevice", serialDevice, *m_config, "Serial");
    std::cerr << m_serialDevice->getValue() << std::endl;
    m_rotator->setBounds(0, 360);

    VrmlNamespace::addBuiltIn(VrmlNode::defineType<VrmlNodeDrehgeber>());

    m_rotator->setCallback([this](float angle, bool) {
        for (auto drehgeber : drehgebers)
            drehgeber->setAngle(angle / 360 * 2 * M_PI);
        });
    auto value = m_delay->getValue();
    m_delay->ui()->setText("delay in ms");
    m_delay->ui()->setBounds(0, 1000);
    m_delay->setValue(value);

    auto updater = [this]() {
    AVRClose();
    std::cerr << "m_serialDevice->getValue()" << m_serialDevice->getValue() << std::endl;
    AVRInit(m_serialDevice->getValue().c_str(), std::stoi(m_baudrate->getValue()));
    };
    m_serialDevice->setUpdater(updater);
    m_baudrate->setUpdater(updater);
    updater();
    start(); // start serial thread
}

Drehgeber::~Drehgeber()
{
    m_config->save();
    running = false;
    AVRClose();
}
bool Drehgeber::update()
{

    if (counter < 0)
        counter += 1000;
    float na = (float)((counter / 1000.0) * 2 * M_PI)-0.6;
    if (na < 0)
        na += 2 * M_PI;

    auto frametime = cover->frameTime();
    m_values.emplace_front(std::make_pair(frametime, na));
    na = m_values.back().second;
    
    while(m_values.back().first < frametime - m_delay->getValue() / 1000)
        m_values.pop_back();

    if (na != angle)
    {
        fprintf(stderr, "Angle %f\n",na); 
        m_rotator->setValue((na/M_PI*360.0/2));
        angle = na;

        for (auto drehgeber : drehgebers)
            drehgeber->setAngle(angle);
        return true;
    }
    return false; // don't request that scene be re-rendered
}

void Drehgeber::run()
{
    while (running)
    {
        int nc = 0;
        unsigned char buf[101];
        unsigned char c;
        do {
            if (!running)
            {
                return;
            }
            bool res = AVRReadBytes(1, &c);
            if (res)
            {
                buf[nc] = c;
                nc++;
                buf[nc] = '\0';
            }
        } while (c != '\n' && nc < 100);
        if (nc > 0)
        {
            sscanf((char*)buf, "%d", &counter);
        }
    }
}