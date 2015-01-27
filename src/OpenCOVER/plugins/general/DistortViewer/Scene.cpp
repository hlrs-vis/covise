/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Scene.h"
#include "ScreenDome.h"
#include "ScreenCylinder.h"
#include "ScreenPlane.h"
#include "ScreenLoad.h"
#include "XmlTools.h"
#include "HelpFuncs.h"
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>

using namespace covise;
using namespace opencover;

bool Scene::visStatus;
Screen *Scene::screen;
std::vector<Projector_ptr> Scene::projectors;

Scene::Scene(bool load, bool status)
{
    visStatus = status;
    if (load)
        loadFromXML();
    else
        init();
}

Scene::~Scene(void)
{
}

void Scene::init()
{
    setScreenShape("Dome");
    makeNewProjector();
}

void Scene::loadFromXML()
{
    resetSceneContent();

    std::string section;
    std::string var_str;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();

    //Screen erstellen
    section = "Geometry";
    std::string screenType = XmlTools::getInstance()->loadStrValue(plugPath + "." + section, "Shape", "Dome");
    setScreenShape(screenType);
    screen->loadFromXML();

    //Anzahl der configurierten Projectors auslesen und erstellen
    section = "Projectors";
    int projCount = XmlTools::getInstance()->loadIntValue(plugPath + "." + section, "ProjCount", 0);
    if (projCount > 0)
    {
        for (unsigned int i = 0; i < projCount; i++)
        {
            Projector_ptr projector(new Projector(true));
            projectors.push_back(projector);
        }
    }
    else
        makeNewProjector();
}

void Scene::saveToXML()
{
    std::string section;
    std::string var_str;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();

    //Screen abspeichern
    screen->saveToXML();

    //Projectors
    section = "Projectors";
    XmlTools::getInstance()->saveIntValue(projectors.size(), plugPath + "." + section, "ProjCount");
    for (unsigned int i = 0; i < projectors.size(); i++)
    {
        projectors.at(i)->saveToXML();
    }

    //Eintr�ge in xml schreiben
    XmlTools::getInstance()->saveToXml();
}

void Scene::resetSceneContent()
{
    //Projectors-Array leeren
    projectors.clear();
}

bool Scene::setScreenShape(std::string shapeName)
{
    bool error = false;

    delete screen;
    if (!shapeName.empty())
    {
        if (strcasecmp(shapeName.c_str(), "Dome") == 0)
            screen = new ScreenDome();
        else
        {
            if (strcasecmp(shapeName.c_str(), "Cylinder") == 0)
                screen = new ScreenCylinder();
            else
            {
                if (strcasecmp(shapeName.c_str(), "Plane") == 0)
                    screen = new ScreenPlane();
                else
                {
                    if (strcasecmp(shapeName.c_str(), "Custom Geometry") == 0)
                        screen = new ScreenLoad();
                    else
                    {
                        std::cerr << "Geometry of type " << shapeName.c_str() << " is unknown!" << std::endl;
                        error = true;
                    }
                }
            }
        }
    }
    else
        error = true;

    if (error)
    {
        screen = new ScreenDome();
        std::cerr << "Default geometry has been set to: Dome\n" << std::endl;
        return false;
    }
    else
        return true;
}

Projector *Scene::getProjector(int no)
{
    return projectors.at(no).get();
}

void Scene::makeNewProjector()
{
    Projector_ptr projector(new Projector());
    projectors.push_back(projector);
    std::cerr << "New Projector [number " << projectors.size() << "] added!" << std::endl;
}

void Scene::deleteProjector(int delNum)
{
    //Iterator der auf zu loeschenden Projektoreintrag zeigt (zaehler startet bei 0)
    vector<Projector_ptr>::iterator itRemove = projectors.begin() + (delNum);
    //Eintrag Loeschen
    projectors.erase(itRemove);

    //Projektornummern (start bei 1) neu setzen (->keine Luecken)
    for (unsigned int i = 0; i < projectors.size(); i++)
    {
        projectors.at(i)->setProjectorNum(i + 1);
    }
}

int Scene::getNumProjectors()
{
    return projectors.size();
}
