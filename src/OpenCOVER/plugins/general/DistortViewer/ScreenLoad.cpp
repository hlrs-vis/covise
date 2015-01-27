/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ScreenLoad.h"
#include "XmlTools.h"
#include <cover/coVRFileManager.h>

#include <osgDB/ReadFile>

ScreenLoad::ScreenLoad(void)
{
}

ScreenLoad::~ScreenLoad(void)
{
}

void ScreenLoad::loadFromXML()
{
    Screen::loadFromXML();

    std::string section = "Geometry";
    std::string subsection = geoShape;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();
    std::string path = plugPath + "." + section + "." + subsection;

    fileName = XmlTools::getInstance()->loadStrValue(path, "fileName", "");
}

void ScreenLoad::saveToXML()
{
    Screen::saveToXML();

    std::string var_str;
    std::string section = "Geometry";
    std::string subsection = geoShape;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();
    std::string path = plugPath + "." + section + "." + subsection;

    XmlTools::getInstance()->saveStrValue(fileName, path, "fileName");

    //Einträge in xml schreiben
    XmlTools::getInstance()->saveToXml();
}

bool ScreenLoad::fileIsValid()
{
    bool valid = fileIsValid(fileName);
    return valid;
}

bool ScreenLoad::fileIsValid(std::string file)
{
    const char *filepath = coVRFileManager::instance()->getName(file.c_str());
    if (filepath == NULL)
    {
        std::cerr << "Geometry-File: " << file.c_str() << " not found!\n" << std::endl;
        return false;
    }

    osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(filepath);
    if (model == NULL)
    {
        std::cerr << "Geometry-File: " << file.c_str() << " could not load!\n" << std::endl;
        return false;
    }
    osg::ref_ptr<osg::Geode> modelGeode = model->asGeode();
    if (modelGeode == NULL)
    {
        std::cerr << "Geometry-File must have a GeoNode as root to be loaded!\n" << std::endl;
        return false;
    }
    return true;
}

void ScreenLoad::setFilename(std::string new_fileName)
{
    fileName = new_fileName;
}

osg::Geode *ScreenLoad::drawScreen(bool gitter)
{
    const char *filepath = coVRFileManager::instance()->getName(fileName.c_str());
    if (filepath == NULL)
    {
        std::cerr << "Geometry-File: " << filepath << " not found!\n" << std::endl;
        return NULL;
    }
    osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(filepath);
    if (model == NULL)
        std::cerr << "Geometry-File: " << fileName.c_str() << " could not load!\n" << std::endl;
    osg::ref_ptr<osg::Geode> modelGeode = model->asGeode();
    if (modelGeode == NULL)
        std::cerr << "Geometry-File must have a GeoNode as root to be loaded!\n" << std::endl;

    return modelGeode;
}
