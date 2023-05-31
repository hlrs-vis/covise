/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

//
//

#include "XElevator.h"
#include "XCar.h"
#include "XLanding.h"
#include <cover/coVRTui.h>

#include <net/covise_host.h>
#include <net/covise_socket.h>

using namespace covise;


static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeXElevator(scene);
}

// Define the built in VrmlNodeType:: "XElevator" fields

VrmlNodeType *VrmlNodeXElevator::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("XElevator", creator);
    }

    VrmlNodeGroup::defineType(t); // Parent class
    t->addExposedField("landingHeights", VrmlField::MFFLOAT);
    t->addExposedField("shaftPositions", VrmlField::MFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodeXElevator::nodeType() const
{
    return defineType(0);
}

VrmlNodeXElevator::VrmlNodeXElevator(VrmlScene *scene)
    : VrmlNodeGroup(scene)
{
    setModified();
    XElevatorTab = new coTUITab("XElevator", coVRTui::instance()->mainFolder->getID());
    XElevatorTab->setPos(0, 0);
}

VrmlNodeXElevator::VrmlNodeXElevator(const VrmlNodeXElevator &n)
    : VrmlNodeGroup(n.d_scene)
{
    setModified();
    XElevatorTab = n.XElevatorTab;
}

VrmlNodeXElevator::~VrmlNodeXElevator()
{
}

VrmlNode *VrmlNodeXElevator::cloneMe() const
{
    return new VrmlNodeXElevator(*this);
}

VrmlNodeXElevator *VrmlNodeXElevator::toXElevator() const
{
    return (VrmlNodeXElevator *)this;
}

ostream &VrmlNodeXElevator::printFields(ostream &os, int indent)
{

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeXElevator::setField(const char *fieldName,
                                const VrmlField &fieldValue)
{
    if
        TRY_FIELD(landingHeights, MFFloat)
    else if
        TRY_FIELD(shaftPositions, MFFloat)
    else
    VrmlNodeGroup::setField(fieldName, fieldValue);
    
    if(strcmp(fieldName,"children")==0)
    {
        
        stations.resize(d_landingHeights.size()*d_shaftPositions.size());
        Landings.resize(d_landingHeights.size()*d_shaftPositions.size());
        for(int i=0;i<stations.size();i++)
        {
            stations[i].Car=NULL;
            Landings[i]=NULL;
        }
        for(int i=0;i<d_shaftPositions.size();i++)
        {
            shafts.push_back(new Shaft);
        }

        for(int i=0;i<d_children.size();i++)
        {
            VrmlNodeXCar *XCar = dynamic_cast<VrmlNodeXCar *>(d_children[i]);
            if(XCar)
            {
                if(XCar->d_carNumber.get() >= Cars.size())
                {
                    Cars.resize(XCar->d_carNumber.get()+1);
                }
                Cars[XCar->d_carNumber.get()] = XCar;
                XCar->setElevator(this);
            }
            VrmlNodeXLanding *Landing = dynamic_cast<VrmlNodeXLanding *>(d_children[i]);
            if(Landing)
            {
                if(Landing->d_LandingNumber.get() >= Landings.size())
                {
                    int oldSize=Landings.size();
                    int newSize=Landing->d_LandingNumber.get()+1;
                    Landings.resize(newSize);
                    for(int i=oldSize;i<newSize;i++)
                    {
                        Landings[i]=NULL;
                    }
                }
                Landings[Landing->d_LandingNumber.get()] = Landing;
                Landing->setElevator(this);
            }
        }
        //assign Cars to landing
        for(int i=0;i<Cars.size();i++)
        {
            if(Cars[i])
            {
                int landing = Cars[i]->d_currentLanding.get();
                if(Landings.size() > landing && Landings[landing] !=NULL)
                {
                    Landings[landing]->setCar(Cars[i]);
                }
            }
        }
        
        for(int i=0;i<stations.size();i++)
        {
            
            int Landing = i % d_landingHeights.size();
            int shaft = i / d_landingHeights.size();
            stations[i].setX(d_shaftPositions[shaft]);
            stations[i].setY(d_landingHeights[Landing]);
        }
    }
}

const VrmlField *VrmlNodeXElevator::getField(const char *fieldName)
{
    if (strcmp(fieldName, "LandingHeights") == 0)
        return &d_landingHeights;
    else if (strcmp(fieldName, "shaftPositions") == 0)
        return &d_shaftPositions;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeXElevator::eventIn(double timeStamp,
                               const char *eventName,
                               const VrmlField *fieldValue)
{

    // Check exposedFields
    //else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

void VrmlNodeXElevator::render(Viewer *)
{
    
    for(int i=0;i<Cars.size();i++)
    {
        VrmlNodeXCar *XCar = Cars[i];
        if(XCar!=NULL)
        {
			
            if((XCar->getState()==VrmlNodeXCar::Idle))
            {
                {
                    // tell it to move to next stop
                    XCar->moveToNext();
                }
            }
            XCar->update();
        }
    }

}


bool VrmlNodeXElevator::occupy(int landing,VrmlNodeXCar *Car)
{
    bool success=false;
    if(stations[landing].Car == NULL || stations[landing].Car == Car)
    {
        success = true;
        stations[landing].Car = Car;
    }
    else
    {
        return false;
    }
    
    if(Landings.size() > landing && Landings[landing] !=NULL)
    {
        if(Landings[landing]->getCar() != NULL && Landings[landing]->getCar() != Car)
            return false;
        Landings[landing]->setCar(Car);
        success=true;
    }
    return success; // no Landing
}
void VrmlNodeXElevator::release(int landing)
{
    if(Landings.size() > landing && Landings[landing] !=NULL)
    {
        Landings[landing]->setCar(NULL);
    }
}

void VrmlNodeXElevator::goTo(int landing)
{
  /*  if (Cars[0]->d_currentStationIndex.get() == landing)
    {
        Cars[0]->
    }*/
    Cars[0]->goTo(landing);
}

XElevatorPlugin* XElevatorPlugin::plugin = nullptr;


XElevatorPlugin::XElevatorPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "ElevatorPlugin::ElevatorPlugin\n");

    plugin = this;
}

// this is called if the plugin is removed at runtime
XElevatorPlugin::~XElevatorPlugin()
{
    fprintf(stderr, "ElevatorPlugin::~ElevatorPlugin\n");

}

bool XElevatorPlugin::init()
{
    VrmlNamespace::addBuiltIn(VrmlNodeXElevator::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeXCar::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeXLanding::defineType());

    return true;
}

bool
XElevatorPlugin::update()
{
    return false;
}

COVERPLUGIN(XElevatorPlugin)