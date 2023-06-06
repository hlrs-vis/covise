#include "Drehgeber.h"

#include <cover/coVRPluginSupport.h>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlNodeTransform.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlSFFloat.h>

#include <util/coExport.h>
#include <vector>
using namespace vrml;


class VrmlNodeDrehgeber;
std::vector<VrmlNodeDrehgeber*> drehgebers;

class PLUGINEXPORT VrmlNodeDrehgeber : public vrml::VrmlNodeChild
{
public:
    static VrmlNode* creator(VrmlScene* scene)
    {
        return new VrmlNodeDrehgeber(scene);
    }
    VrmlNodeDrehgeber(VrmlScene* scene) : VrmlNodeChild(scene), m_index(drehgebers.size())
    {

        std::cerr << "vrml Machine node created" << std::endl;
        drehgebers.push_back(this);
    }
    ~VrmlNodeDrehgeber()
    {
        drehgebers.erase(drehgebers.begin() + m_index);
    }

    // Define the fields of XCar nodes
    static VrmlNodeType* defineType(VrmlNodeType* t = 0)
    {
        static VrmlNodeType* st = 0;

        if (!t)
        {
            if (st)
                return st; // Only define the type once.
            t = st = new VrmlNodeType("Drehgeber", creator);
        }

        VrmlNodeChild::defineType(t); // Parent class

        t->addEventOut("angle", VrmlField::SFFLOAT);

        return t;
    }
    virtual VrmlNodeType* nodeType() const { return defineType(); };
    VrmlNode* cloneMe() const
    {
        return new VrmlNodeDrehgeber(*this);
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
, m_menu(new ui::Menu("Drehgeber", this))
, m_rotator(new ui::Slider(m_menu, "rotation_angle_in_degree")) 
{
    m_rotator->setBounds(0, 360);
    serialDeviceUI = new ui::TextField(m_menu, "serialDevice");
    baudrateUI = new ui::EditField(m_menu, "baudrate");

    VrmlNamespace::addBuiltIn(VrmlNodeDrehgeber::defineType());

    m_rotator->setCallback([this](float angle, bool) {
        for (auto drehgeber : drehgebers)
            drehgeber->setAngle(angle / 360 * 2 * M_PI);
        });
}
bool Drehgeber::init()
{
    SerialDevice = configString("Serial", "device", "\\\\.\\COM12");
    int64_t br = 115200;
    baudrate = configInt("Serial", "baudrate", br);
    SerialDevice->setUpdater([this](std::string val) {
        AVRClose();
        AVRInit(val.c_str(), (int)*baudrate);
        });
    baudrate->setUpdater([this](int64_t val) {
        AVRClose();
        std::string name = *SerialDevice;
        if (name != serialDeviceUI->value())
            serialDeviceUI->setValue(name);
        AVRInit(name.c_str(), (int)*baudrate);
        });
    serialDeviceUI->setValue(*SerialDevice);
    serialDeviceUI->setCallback([this](std::string dev) {
        if (SerialDevice->value() != dev)
        {
            *SerialDevice = dev;
            config()->save();
        }
        });
    baudrateUI->setValue(br);
    baudrateUI->setCallback([this](const std::string& b) {
        if (baudrate->value() != (int64_t)baudrateUI->number())
        {
            *baudrate = (int64_t)baudrateUI->number();
            config()->save();
        }
        });
    // already initialized above twice std::string name = *SerialDevice;
    //AVRInit(name.c_str(), (int)*baudrate);

    start(); // start serial thread
    return true;
}
Drehgeber::~Drehgeber()
{
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