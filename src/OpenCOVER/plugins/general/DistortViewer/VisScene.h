/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include <osg/Version>
#include <osg/Node>
#include <osg/Camera>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osg/Program>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/TexGen>
#include <osg/TexEnv>
#include <osg/TexMat>
#include <osg/StateSet>

#include <osgDB/WriteFile>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>

#include <cmath>
#include <iostream>

#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
using namespace covise;
using namespace opencover;

class Screen;
class Projector;

class VisScene
{
public:
    VisScene(Projector *c_projector, bool load = false);
    ~VisScene(void);

    osg::Group *getSceneGroup()
    {
        return visGroup.get();
    };
    osg::MatrixTransform *getScreenGeode()
    {
        return geodeScreen.get();
    };
    std::string getBlendImgFilePath()
    {
        return blendImgFilePath;
    };
    void setBlendImgFilePath(std::string filePath)
    {
        blendImgFilePath = filePath;
    };
    bool getDistortState()
    {
        return distort;
    };
    void setDistortState(bool state)
    {
        distort = state;
    };
    bool getBlendState()
    {
        return blend;
    };
    void setBlendState(bool state)
    {
        blend = state;
    };
    void updateVisGroup();
    void updateViewerPos();

    void saveToXML();
    bool loadFromXML();

private:
    osg::Shader *loadShader(const std::string &ShaderFile);

    /** Blau-Textur erstellen und zurückgeben
	 *
	 * @return Blau-Textur
	 */
    osg::Texture2D *makeTexB(void);

    osg::TexMat *makeTexMatB();

    /** Rot/Grün-Textur erstellen und zurückgeben
	 *
	 * @return Rot/Grün-Textur
	 */
    osg::Texture2D *makeTexRG(void);

    osg::TexMat *makeTexMatRG();

    osg::Camera *makeSceneCam(void);
    osg::Camera *makeVisCam(void);
    osg::Group *makeVisGroup();

    //Variablen dekleration
    //----------------------
    Projector *projector;

    bool distort; //Bild verzerrung aktivieren? ja/nein
    bool blend; //Edge-Blending aktivieren? ja/nein
    int visResolutionW; //Auflösung Breite der verzerrten Szene
    int visResolutionH; //Auflösung Höhe der verzerrten Szene
    std::string blendImgFilePath; //Dateipfad zur Image-Datei
    std::string blendImgFile; //rel. Dateipfad des Edgeblending-Images
    std::string vertShaderFile; //rel. Dateipfad zur Vertex-Shader Datei
    std::string fragShaderFile; //rel. Dateipfad zur Fragment Shader Datei

    //Render-Target

    osg::Camera::RenderTargetImplementation renderImplementation;

    osg::ref_ptr<osg::MatrixTransform> geodeScreen; //VektorArray der Screengeometrieen
    osg::ref_ptr<osg::Group> visGroup; //VectorArray der Visualisierungsgruppen

    osg::Matrix coViewMat; //Viewing Matrix von opencover VRViewer
    osg::Matrix coProjMat; //Projection Matrix von opencover VRViewer

    osg::Matrix rotate;
};
