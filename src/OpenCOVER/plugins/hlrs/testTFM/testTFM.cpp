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


class Drehgeber;
std::vector<Drehgeber *> drehgebers;

class PLUGINEXPORT Drehgeber : public vrml::VrmlNodeChild
{
public:
    static VrmlNode *creator(VrmlScene *scene)
    {
        return new Drehgeber(scene);
    }
    Drehgeber(VrmlScene *scene) : VrmlNodeChild(scene), m_index(drehgebers.size())
    {

        std::cerr << "vrml Machine node created" << std::endl;
        drehgebers.push_back(this);
    }
    ~Drehgeber()
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
        return new Drehgeber(*this);
    }
    void setAngle(VrmlSFFloat position)
    {
        auto t = System::the->time();
        eventOut(t, "angle", position);
    }

private:
    size_t m_index = 0;
};



COVERPLUGIN(TestTfm)

TestTfm::TestTfm()
: ui::Owner("TestTfm_owner", cover->ui)
, m_menu(new ui::Menu("TestTfm", this))
, m_rotator(new ui::Slider(m_menu, "rotation_angle_in_degree")) {
    m_rotator->setBounds(0, 360);

    VrmlNamespace::addBuiltIn(Drehgeber::defineType());

    m_rotator->setCallback([this](float angle, bool){
        for(auto drehgeber : drehgebers)
            drehgeber->setAngle(angle /360 * 2 * M_PI);
    });
}