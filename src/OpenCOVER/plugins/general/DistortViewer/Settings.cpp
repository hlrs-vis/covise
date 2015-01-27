/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Settings.h"
#include "XmlTools.h"
#include "HelpFuncs.h"

Settings::Settings(void)
    : visResolutionH(1024)
    , visResolutionW(1024)
    , section("Settings")
    , plugPath(XmlTools::getInstance()->getPlugPath())

{
    path = plugPath + "." + section;
}

Settings::~Settings(void)
{
}

Settings *Settings::getInstance()
{
    static Settings theInstance;
    return &theInstance;
}

void Settings::saveToXML()
{
    XmlTools::getInstance()->saveIntValue(visResolutionW, path, "ResolutionW");
    XmlTools::getInstance()->saveIntValue(visResolutionH, path, "ResolutionH");
    XmlTools::getInstance()->saveStrValue(imagePath, path, "ImagePath");
    XmlTools::getInstance()->saveStrValue(fragShaderFile, path, "FragShaderFile");
    XmlTools::getInstance()->saveStrValue(vertShaderFile, path, "VertShaderFile");
}

void Settings::loadFromXML()
{
    //Auflösung der Visualisierung
    visResolutionW = XmlTools::getInstance()->loadIntValue(path, "ResolutionW", 1024);
    std::cerr << "Aufloesung Breite: " << visResolutionW << "\n" << std::endl;

    visResolutionH = XmlTools::getInstance()->loadIntValue(path, "ResolutionH", 1024);
    std::cerr << "Aufloesung Hoehe: " << visResolutionH << "\n" << std::endl;

    //Pfad für Blend-Images
    imagePath = XmlTools::getInstance()->loadStrValue(path, "ImagePath", "bitmaps/Distortion/");
    std::cerr << "Pfad fuer Bilder: " << imagePath.c_str() << "\n" << std::endl;

    // Pfade für Shader
    fragShaderFile = XmlTools::getInstance()->loadStrValue(path, "FragShaderFile", "share/covise/materials/Cg/Distortion/shader.frag");
    vertShaderFile = XmlTools::getInstance()->loadStrValue(path, "VertShaderFile", "share/covise/materials/Cg/Distortion/shader.vert");
    std::cerr << "shaderFile " << fragShaderFile << std::endl;
}
