#include "testTFM.h"

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
    static VrmlNode *creator(VrmlScene *scene)
    {
        return new VrmlNodeDrehgeber(scene);
    }
    VrmlNodeDrehgeber(VrmlScene *scene) : VrmlNodeChild(scene), m_index(drehgebers.size())
    {

        std::cerr << "vrml Machine node created" << std::endl;
        drehgebers.push_back(this);
    }
    ~VrmlNodeDrehgeber()
    {
        drehgebers.erase(drehgebers.begin() + m_index);
    }

    // Define the fields of XCar nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0)
    {
        static VrmlNodeType *st = 0;

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
    virtual VrmlNodeType *nodeType() const { return defineType(); };
    VrmlNode *cloneMe() const
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
: ui::Owner("TestTfm_owner", cover->ui)
, m_menu(new ui::Menu("TestTfm", this))
, m_rotator(new ui::Slider(m_menu, "rotation_angle_in_degree")) {
    m_rotator->setBounds(0, 360);


    VrmlNamespace::addBuiltIn(VrmlNodeDrehgeber::defineType());

    m_rotator->setCallback([this](float angle, bool){
        for(auto drehgeber : drehgebers)
            drehgeber->setAngle(angle /360 * 2 * M_PI);
    });
}
bool Drehgeber::init()
{
    SerialDevice = configString("Serial", "device", "COM2");
    int64_t br = 115200;
    baudrate = configInt("Serial", "baudrate", br);
    SerialDevice->setUpdater([this](std::string val) {
        AVRClose();
        AVRInit(val.c_str(), (int)baudrate.get()->value());
        });
    baudrate->setUpdater([this](int64_t val) {
        AVRClose();
        AVRInit(SerialDevice.get()->value().c_str(), (int)baudrate.get()->value());
        });

    AVRInit(SerialDevice.get()->value().c_str(), (int)baudrate.get()->value());
    return true;
}
Drehgeber::~Drehgeber()
{
    AVRClose();
}
bool Drehgeber::update()
{
    float na = (float)((counter / 1000.0) * 2*M_PI);
    if (na != angle)
    {
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
        int nc=0;
        unsigned char buf[101];
        unsigned char c;
        do {
        AVRReadBytes(1,&c);
        buf[nc] = c;
        nc++;
        buf[nc] = '\0';
        } while (c != '\n' && nc < 100);
        sscanf((char *)buf, "%d", &counter);
    }
}