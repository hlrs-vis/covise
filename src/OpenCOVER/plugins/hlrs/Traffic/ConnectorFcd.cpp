#include "ConnectorFcd.h"

#include <cover/coHud.h>

#include <algorithm>
#include <cover/OpenCOVER.h>
#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMDocumentType.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/dom/DOMLSOutput.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMNodeIterator.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMText.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/framework/MemBufFormatTarget.hpp>
#include <xercesc/framework/MemBufInputSource.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XercesVersion.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/validators/common/Grammar.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/util/XMLString.hpp>

#include <cover/coVRMSController.h>

#include "Traffic.h"

using namespace xercesc;

ConnectorFcd::ConnectorFcd(const std::string &filename)
{
    XMLPlatformUtils::Initialize();

    TAG_root = XMLString::transcode("fcd-export");
    TAG_timestep = XMLString::transcode("timestep");
    TAG_vehicle = XMLString::transcode("vehicle");
    ATTR_time = XMLString::transcode("time");
    ATTR_id = XMLString::transcode("id");
    ATTR_type = XMLString::transcode("type");
    ATTR_x = XMLString::transcode("x");
    ATTR_y = XMLString::transcode("y");
    ATTR_z = XMLString::transcode("z");
    ATTR_angle = XMLString::transcode("angle");
    ATTR_speed = XMLString::transcode("speed");

    SAX2XMLReader *parser = XMLReaderFactory::createXMLReader();
    parser->setExitOnFirstFatalError(true);

    parser->setContentHandler(this);
    parser->setErrorHandler(this);

    parser->parse(filename.c_str());
}

double getDouble(const Attributes &attrs, XMLCh *attr)
{
    auto idx = attrs.getIndex(attr);
    if (idx == -1)
        return 0.0;

    auto value_str = XMLString::transcode(attrs.getValue(idx));
    double value = std::atof(value_str);
    XMLString::release(&value_str);

    return value;
}

std::string getString(const Attributes &attrs, XMLCh *attr)
{
    auto idx = attrs.getIndex(attr);
    if (idx == -1)
        return "";

    auto value_str = XMLString::transcode(attrs.getValue(idx));
    std::string value(value_str);
    XMLString::release(&value_str);

    return std::move(value);
}

double sumoAngleToMath2(double angle)
{
    // See https://github.com/eclipse-sumo/sumo/issues/1372
    // Return a normal math angle that works with sin/cos, i.e. radians, counter-clockwise, from positive X-axis.
    return ((90.0 - angle) / 180.0) * M_PI;
}

void ConnectorFcd::startElement(const XMLCh *const uri,
    const XMLCh *const localname,
    const XMLCh *const qname,
    const Attributes &attrs)
{
    if (XMLString::equals(localname, TAG_timestep))
    {
        double time = getDouble(attrs, ATTR_time);

        if (m_parseTimestep >= 0)
        {
            // copy old state
            m_simulationStates[time] = m_simulationStates[m_parseTimestep];
        }

        m_parseTimestep = time;
        m_timesteps.insert(time);

        if (m_timesteps.size() % 1000 == 0)
        {
            opencover::OpenCOVER::instance()->hud->setText3("Loading Traffic Simulation...");
            opencover::OpenCOVER::instance()->hud->redraw();
        }
    }
    else if (XMLString::equals(localname, TAG_vehicle))
    {
        vehicle_id_t id(getString(attrs, ATTR_id));
        vehicle_class_t vehicleClass(getString(attrs, ATTR_type));
        double x = getDouble(attrs, ATTR_x);
        double y = getDouble(attrs, ATTR_y);
        double z = getDouble(attrs, ATTR_z);
        double speed = getDouble(attrs, ATTR_speed);
        double angle = getDouble(attrs, ATTR_angle);

        auto &state = m_simulationStates[m_parseTimestep];
        state.vehicles[id] = VehicleState {
            id,
            vehicleClass,
            osg::Vec3d(x, y, z),
            sumoAngleToMath2(angle),
            speed,
        };
    }
}

void ConnectorFcd::fatalError(const SAXParseException &exception)
{
    char *message = XMLString::transcode(exception.getMessage());
    std::cout << "Fatal Error: " << message
              << " at line: " << exception.getLineNumber()
              << std::endl;
    XMLString::release(&message);
}

ConnectorFcd::~ConnectorFcd()
{
    XMLString::release(&TAG_root);
    XMLString::release(&TAG_timestep);
    XMLString::release(&TAG_vehicle);
    XMLString::release(&ATTR_time);
    XMLString::release(&ATTR_id);
    XMLString::release(&ATTR_type);
    XMLString::release(&ATTR_x);
    XMLString::release(&ATTR_y);
    XMLString::release(&ATTR_z);
    XMLString::release(&ATTR_angle);
    XMLString::release(&ATTR_speed);

    XMLPlatformUtils::Terminate();
}

bool ConnectorFcd::isConnected() const
{
    return true;
}

bool ConnectorFcd::update(double deltaTime, double simulationDeltaTime)
{
    m_simulationTime += simulationDeltaTime;
    auto it = std::find_if(m_timesteps.begin(), m_timesteps.end(), [&](double t)
        { return t >= m_simulationTime; });

    if (it == m_timesteps.end() || *it == m_lastSimulationTime)
        return false;

    m_lastSimulationTime = *it;
    return true;
}

void ConnectorFcd::getSimulationState(SimulationState &state)
{
    if (m_lastSimulationTime >= 0.0)
    {
        state = m_simulationStates[m_lastSimulationTime];
    }
}
