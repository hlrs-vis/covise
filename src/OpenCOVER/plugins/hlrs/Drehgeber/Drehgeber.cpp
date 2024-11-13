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
    
    static const char* name() { return "Drehgeber"; }

    VrmlNodeDrehgeber(VrmlScene* scene) : VrmlNodeChild(scene, name()), m_index(drehgebers.size())
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



COVERPLUGIN(Drehgeber)

Drehgeber::Drehgeber()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("Drehgeber_owner", cover->ui)
, m_config(config())
, m_menu(new ui::Menu("Drehgeber", this))
, m_rotator(new ui::Slider(m_menu, "rotation_angle_in_degree")) 
, m_serialDevice(std::make_unique<ui::EditFieldConfigValue>(m_menu, "serialDevice", "\\\\.\\COM12", *m_config, "Serial")) 
, m_baudrate(std::make_unique<ui::EditFieldConfigValue>(m_menu, "baudrate", "115200", *m_config, "Serial")) 
, m_delay(std::make_unique<ui::SliderConfigValue>(m_menu, "delay", 0, *m_config, "Serial")) 
{
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