/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Projector.h"
#include "Scene.h"
#include "Screen.h"
#include "ScreenPlane.h"
#include "HelpFuncs.h"
#include "XmlTools.h"
#include "Settings.h"

#include "KoordAxis.h"

#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <cover/coVRFileManager.h>
#include <sstream>

VisScene::VisScene(Projector *c_projector, bool load)
    : projector(c_projector)
    , blend(false)
    , distort(false)
    , visResolutionH(Settings::getInstance()->visResolutionH)
    , visResolutionW(Settings::getInstance()->visResolutionW)
{
    coViewMat = coVRConfig::instance()->channels[projector->getProjectorNum() - 1].rightView;
    coProjMat = coVRConfig::instance()->channels[projector->getProjectorNum() - 1].rightProj;

    //KS-Transformation von VRViewer ViewMat (OpenGL) in OSG
    rotate.makeRotate(osg::PI_2, osg::X_AXIS, //90Grad um x-Achse
                      0, osg::Y_AXIS,
                      0, osg::Z_AXIS);

    if (load)
        loadFromXML();

    visGroup = makeVisGroup();
}

VisScene::~VisScene(void)
{
}

void VisScene::saveToXML()
{
    std::string section = "VisScenes";
    std::string plugPath = XmlTools::getInstance()->getPlugPath();
    std::string num_str;
    HelpFuncs::IntToString(projector->getProjectorNum(), num_str);
    std::string path = plugPath + ".Proj" + num_str + "." + section;

    XmlTools::getInstance()->saveBoolValue(distort, path, "Distort");
    XmlTools::getInstance()->saveBoolValue(blend, path, "Blend");
    XmlTools::getInstance()->saveStrValue(blendImgFilePath, path, "BlendImg");
}

bool VisScene::loadFromXML()
{
    std::string section = "VisScenes";
    std::string plugPath = XmlTools::getInstance()->getPlugPath();
    std::string num_str;
    HelpFuncs::IntToString(projector->getProjectorNum(), num_str);
    std::string path = plugPath + ".Proj" + num_str + "." + section;

    distort = XmlTools::getInstance()->loadBoolValue(path, "Distort", true);
    std::cerr << "Distortion for Projector " << num_str << ((distort) ? " enabled" : " disabled") << "\n" << std::endl;

    blend = XmlTools::getInstance()->loadBoolValue(path, "Blend", false);
    std::cerr << "Blending for Projector " << num_str << ((blend) ? " enabled" : " disabled") << "\n" << std::endl;

    // Pfad für Blend-Image aus Config auslesen
    if (blend)
    {
        std::string imagePath = Settings::getInstance()->imagePath;
        blendImgFilePath = XmlTools::getInstance()->loadStrValue(path, "BlendImg", imagePath + "\blendProj" + num_str + ".png");
        const char *filepath = coVRFileManager::instance()->getName((blendImgFilePath).c_str());
        if ((filepath != NULL) & blend)
            blendImgFile = filepath;
        else
        {
            std::cerr << "Edge-Blending file " << blendImgFilePath.c_str() << " could not be found in " << imagePath.c_str() << "!\n" << std::endl;
            blend = false;
            return false;
        }
    }

    //Shader Dateien
    if (distort)
    {
        std::string fragShaderFile_gl = Settings::getInstance()->fragShaderFile;
        const char *filepathFrag = coVRFileManager::instance()->getName((fragShaderFile_gl).c_str());
        if (filepathFrag != NULL)
            fragShaderFile = filepathFrag;
        else
        {
            std::cerr << "Fragment-Shader file (" << fragShaderFile_gl.c_str() << ") could not be found!\n" << std::endl;
            distort = false;
            return false;
        }

        std::string vertShaderFile_gl = Settings::getInstance()->vertShaderFile;
        const char *filepathVert = coVRFileManager::instance()->getName((vertShaderFile_gl).c_str());
        if (filepathVert != NULL)
            vertShaderFile = filepathVert;
        else
        {
            std::cerr << "Vertex-Shader file (" << vertShaderFile_gl.c_str() << ") could not be found!\n" << std::endl;
            distort = false;
            return false;
        }
    }

    // Voreingestellte Option für Render-Target aus Config auslesen
    std::string buf = coCoviseConfig::getEntry("COVER.Plugin.Vrml97.RTTImplementation");
    if (!buf.empty())
    {
        if (strcasecmp(buf.c_str(), "fbo") == 0)
            renderImplementation = osg::Camera::FRAME_BUFFER_OBJECT;
        if (strcasecmp(buf.c_str(), "pbuffer") == 0)
            renderImplementation = osg::Camera::PIXEL_BUFFER;
        if (strcasecmp(buf.c_str(), "pbuffer-rtt") == 0)
            renderImplementation = osg::Camera::PIXEL_BUFFER_RTT;
        if (strcasecmp(buf.c_str(), "fb") == 0)
            renderImplementation = osg::Camera::FRAME_BUFFER;
        if (strcasecmp(buf.c_str(), "window") == 0)
            renderImplementation = osg::Camera::SEPERATE_WINDOW;
    }
    else
    {
        renderImplementation = osg::Camera::FRAME_BUFFER_OBJECT;
        buf = "fbo";
    }
    std::cerr << "Set RenderImplementation to: " << buf.c_str() << "\n" << std::endl;

    return true;
}

osg::Shader *VisScene::loadShader(const std::string &ShaderFile)
{
    bool fileFound = osgDB::fileExists(ShaderFile);
    if (fileFound)
    {
        osg::ref_ptr<osg::Shader> shader = osgDB::readShaderFile(ShaderFile);
        if (!shader)
        {
            std::cerr << "Shader-File " << ShaderFile.c_str() << " could not be read!\n" << std::endl;
            return NULL;
        }
        return shader.release();
    }
    std::cerr << "Shader-File " << ShaderFile.c_str() << " could not be found!\n" << std::endl;
    return NULL;
}

osg::Group *VisScene::makeVisGroup()
{
    //Falls keine Scene zur Verzerrung errechnet werden soll -> leere Gruppe zurück
    if (!Scene::getVisStatus())
    {
        osg::ref_ptr<osg::Group> empty_visGroup = new osg::Group();
        return empty_visGroup;
    }

    //-------------------------------
    // ProjektionsGeometrie mit Distortion-Textur erstellen
    //-------------------------------

    geodeScreen = Scene::getScreen()->draw(false);
    geodeScreen->setName("Screen");

    //Statemachine erstellen
    osg::ref_ptr<osg::StateSet> stateScreen = new osg::StateSet();
    geodeScreen->setStateSet(stateScreen.get());

    //TexturRG
    //--------------------
    osg::ref_ptr<osg::Texture2D> textureRG = makeTexRG();
    stateScreen->setTextureAttributeAndModes(0, textureRG.get(), osg::StateAttribute::ON); //(TexUnit, Texturtyp, Attribut)

    //Decal-Mode setzen
    osg::ref_ptr<osg::TexEnv> texEnvRG = new osg::TexEnv();
    texEnvRG->setMode(osg::TexEnv::DECAL);
    stateScreen->setTextureAttributeAndModes(0, texEnvRG.get(), osg::StateAttribute::ON); //(TexUnit, Texturtyp, Attribut)

    //Generiere Texturkoordinaten in x-z-Ebene automatisch
    osg::ref_ptr<osg::TexGen> texGenRG = new osg::TexGen();
    texGenRG->setMode(osg::TexGen::EYE_LINEAR);
    texGenRG->setPlane(osg::TexGen::S, osg::Plane(1.0f, 0.0f, 0.0f, 0.0f));
    texGenRG->setPlane(osg::TexGen::T, osg::Plane(0.0f, 1.0f, 0.0f, 0.0f));
    texGenRG->setPlane(osg::TexGen::R, osg::Plane(0.0f, 0.0f, 1.0f, 0.0f));
    texGenRG->setPlane(osg::TexGen::Q, osg::Plane(0.0f, 0.0f, 0.0f, 1.0f));
    stateScreen->setTextureAttributeAndModes(0, texGenRG.get(), osg::StateAttribute::ON); //(TexUnit, Texturtyp, Attribut)

    //Texturmatrix erstellen und zuweisen
    osg::ref_ptr<osg::TexMat> texMatRG = makeTexMatRG();
    stateScreen->setTextureAttributeAndModes(0, texMatRG.get(), osg::StateAttribute::ON); //(TexUnit, Texturtyp, Attribut)

    //TexturB
    //--------------------
    osg::ref_ptr<osg::Texture2D> textureB = makeTexB();
    stateScreen->setTextureAttributeAndModes(1, textureB.get(), osg::StateAttribute::ON); //(TexUnit, Texturtyp, Attribut)

    //Blend-Mode setzen
    osg::ref_ptr<osg::TexEnv> texEnvB = new osg::TexEnv();
    texEnvB->setMode(osg::TexEnv::BLEND);
    texEnvB->setColor(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f)); //Welche Kanal soll geblendet werden -> hier blau
    stateScreen->setTextureAttributeAndModes(1, texEnvB.get(), osg::StateAttribute::ON); //(TexUnit, Texturtyp, Attribut)

    //Generiere Texturkoordinaten in x-z-Ebene automatisch
    osg::ref_ptr<osg::TexGen> texGenB = new osg::TexGen();
    texGenB->setMode(osg::TexGen::EYE_LINEAR);
    texGenB->setPlane(osg::TexGen::S, osg::Plane(1.0f, 0.0f, 0.0f, 0.0f));
    texGenB->setPlane(osg::TexGen::T, osg::Plane(0.0f, 1.0f, 0.0f, 0.0f));
    texGenB->setPlane(osg::TexGen::R, osg::Plane(0.0f, 0.0f, 1.0f, 0.0f));
    texGenB->setPlane(osg::TexGen::Q, osg::Plane(0.0f, 0.0f, 0.0f, 1.0f));
    stateScreen->setTextureAttributeAndModes(1, texGenB.get(), osg::StateAttribute::ON); //(TexUnit, Texturtyp, Attribut)

    //Texturmatrix erstellen
    osg::ref_ptr<osg::TexMat> texMatB = makeTexMatB();
    stateScreen->setTextureAttributeAndModes(1, texMatB.get(), osg::StateAttribute::ON); //(TexUnit, Texturtyp, Attribut)

    //-------------------------------
    // Kamera mit Projektorsicht erstellen
    //-------------------------------

    osg::ref_ptr<osg::Camera> projCam = projector->getProjCam();
    projCam->setName("projCam");

    //render-Modus festlegen
    projCam->setRenderTargetImplementation(renderImplementation);

    // setze Sichtfenster
    projCam->setViewport(0, 0, visResolutionW, visResolutionH);

    //Screen soll gerendert werden
    projCam->addChild(geodeScreen.get());

    // Bild aus Kameransicht erstellen und speichern
    osg::ref_ptr<osg::Image> distImg = new osg::Image();
    distImg->allocateImage(visResolutionW, visResolutionH, 1, GL_RGBA, GL_UNSIGNED_BYTE);
    projCam->attach(osg::Camera::COLOR_BUFFER, distImg.get());

    //Konfiguration in osgViewer darstellen
    osgViewer::Viewer viewer;
    osg::Group *viewerGroup = new osg::Group();
    viewerGroup->addChild(geodeScreen.get());

    KoordAxis *koordAxis = new KoordAxis();
    osg::MatrixTransform *koordGroup = new osg::MatrixTransform;
    koordGroup->addChild(koordAxis->createAxesGeometry(1000));
    viewerGroup->addChild(koordGroup);

    osg::MatrixTransform *transViewer = new osg::MatrixTransform();
    transViewer->addChild(koordAxis->createAxesGeometry(100));
    transViewer->setMatrix(coViewMat);

    viewerGroup->addChild(transViewer);

    viewer.setSceneData(viewerGroup);
    viewer.setUpViewInWindow(100, 100, 512, 512, 0);
    viewer.run();

    //---------------------------------------
    //Quad zur Ergebisvisualisierung erstellen
    //---------------------------------------

    //Quad Geom und Geode erstellen
    //-------------------------------
    ScreenPlane quad = ScreenPlane(visResolutionW, visResolutionH);

    osg::ref_ptr<osg::Geode> geodeDistQuad = quad.drawScreen(false);
    geodeDistQuad->setName("distQuad");

    // Stateset erstellen und Quad zuweisen
    osg::ref_ptr<osg::StateSet> stateDistQuad = new osg::StateSet();
    stateDistQuad->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    stateDistQuad->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
    geodeDistQuad->setStateSet(stateDistQuad.get());

    //Distortion-Textur (TexUnit 0) erstellen und distQuad zuweisen
    //-----------------------------------------------
    /*osg::ref_ptr<osg::Texture2D> texDist = new osg::Texture2D;
    texDist->setTextureSize(visResolutionW, visResolutionH);
	texDist->setInternalFormat(GL_RGBA);
    texDist->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::NEAREST);
	texDist->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::NEAREST);*/

    //VRViewer legt bei start von OpenCover fest, in welche Textur die Kamera (SceneCam) des zugehörigen projScreens ihre Szene rendert (R2T)
    //Diese Szene soll später verzerrt werden -> Textur mit Szene der TexUnit 0 des Quads zuweisen
    stateDistQuad->setTextureAttributeAndModes(0, coVRConfig::instance()->PBOs[projector->getProjectorNum() - 1].renderTargetTexture, osg::StateAttribute::ON);

    if (blend)
    {
        //Textur für edge-Blending erstellen
        //----------------------------------

        //Edge-Blending File laden
        osg::ref_ptr<osg::Image> blendImg = new osg::Image();
        blendImg->allocateImage(visResolutionW, visResolutionH, 1, GL_RGBA, GL_UNSIGNED_BYTE);
        blendImg = osgDB::readImageFile(blendImgFile);

        //Edge-Blending Textur erstellen
        osg::ref_ptr<osg::Texture2D> texBlendImg = new osg::Texture2D;
        texBlendImg->setTextureSize(visResolutionW, visResolutionH);
        texBlendImg->setInternalFormat(GL_RGBA);
        texBlendImg->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
        texBlendImg->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
        texBlendImg->setImage(blendImg.get());

        //Textur mit EndgeBlending-Image der TexUnit 2 des Quads zuweisen
        stateDistQuad->setTextureAttributeAndModes(2, texBlendImg.get(), osg::StateAttribute::ON);
    }

    //Textur mit DistortionImage erstellen und distQuad zuweisen
    //-----------------------------------------------
    osg::ref_ptr<osg::Texture2D> texDistImg = new osg::Texture2D;
    texDistImg->setTextureSize(visResolutionW, visResolutionH);
    texDistImg->setInternalFormat(GL_RGBA);
    texDistImg->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
    texDistImg->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
    texDistImg->setImage(distImg.get());
    //Textur mit Distortion-Image der TexUnit 1 des Quads zuweisen
    stateDistQuad->setTextureAttributeAndModes(1, texDistImg.get(), osg::StateAttribute::ON);

    // create shaders for distortion
    osg::ref_ptr<osg::StateSet> shaderState = geodeDistQuad->getOrCreateStateSet();
    osg::ref_ptr<osg::Program> distortProgram = new osg::Program;
    distortProgram->setName("shaderDist");
    std::cerr << "vertShaderFile " << vertShaderFile << std::endl;
    std::cerr << "fragShaderFile " << fragShaderFile << std::endl;
    osg::ref_ptr<osg::Shader> shaderVert = loadShader(vertShaderFile);
    osg::ref_ptr<osg::Shader> shaderFrag = loadShader(fragShaderFile);
    if (shaderFrag != NULL && shaderVert != NULL) //Falls Shader-Dateien korrekt geladen
    {
        shaderVert->setType(osg::Shader::VERTEX);
        shaderFrag->setType(osg::Shader::FRAGMENT);
        distortProgram->addShader(shaderVert.get());
        distortProgram->addShader(shaderFrag.get());
        shaderState->addUniform(new osg::Uniform("blend", blend));
        shaderState->addUniform(new osg::Uniform("distort", distort));
        shaderState->addUniform(new osg::Uniform("textureDistImg", 1));
        shaderState->addUniform(new osg::Uniform("textureDistort", 0));
        shaderState->addUniform(new osg::Uniform("textureBlendImg", 2));
        shaderState->setAttributeAndModes(distortProgram.get(), osg::StateAttribute::ON);
    }

    //-----------------------------------
    // VisCam die Quad im Vollbild aufnimmt
    //-----------------------------------
    osg::ref_ptr<osg::Camera> visCam = makeVisCam();
    // add Quad
    visCam->addChild(geodeDistQuad.get());
    visCam->setName("visCam");

    osg::ref_ptr<osg::Group> new_visGroup = new osg::Group();
    new_visGroup->setName("visGroup");
    new_visGroup->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    new_visGroup->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);

    //hinzufügen aller Kameras, damit diese gerendert werden
    new_visGroup->addChild(projCam.get());
    new_visGroup->addChild(visCam.get());

    return new_visGroup.release();
}

void VisScene::updateVisGroup()
{
    osg::ref_ptr<osg::Group> new_visGroup = makeVisGroup();

    for (unsigned int i = 0; i < new_visGroup->getNumChildren(); i++)
    {
        visGroup->replaceChild(visGroup->getChild(i), new_visGroup->getChild(i));
    }
}

void VisScene::updateViewerPos()
{
    //View und Proj-Matrizen des Viewers neu laden
    coViewMat = coVRConfig::instance()->channels[projector->getProjectorNum() - 1].rightView;
    coProjMat = coVRConfig::instance()->channels[projector->getProjectorNum() - 1].rightProj;

    osg::ref_ptr<osg::StateSet> stateScreen = geodeScreen->getStateSet();
    //Texturmatrizen neu erstellen und zuweisen, da sich ViewerPos verändert haben könnte
    osg::ref_ptr<osg::TexMat> texMatB = makeTexMatB();
    stateScreen->setTextureAttributeAndModes(1, texMatB.get(), osg::StateAttribute::ON); //(TexUnit, Texturtyp, Attribut)
    osg::ref_ptr<osg::TexMat> texMatRG = makeTexMatRG();
    stateScreen->setTextureAttributeAndModes(0, texMatRG.get(), osg::StateAttribute::ON); //(TexUnit, Texturtyp, Attribut)
}

osg::Texture2D *VisScene::makeTexRG(void)
{
    // Textur Rot-Grün
    //--------------------------------------------------

    //Bild erstellen, in dem Texturkoordinaten pro Pixel gespeichert werden
    //Rot-, GrünWerte: 2*8Bit
    int imgBreiteRG = 256;
    int imgHoeheRG = 256;
    unsigned char *dataRG;
    dataRG = new unsigned char[imgBreiteRG * imgHoeheRG * 4]; //höhe*breite*farbtiefe[R,G,B,Alpha]
    for (int y = 0; y < imgHoeheRG; ++y)
    {
        for (int x = 0; x < imgBreiteRG; ++x)
        {
            dataRG[x * 4 + y * imgBreiteRG * 4 + 0] = x; //Rot-Wert (8bit)
            dataRG[x * 4 + y * imgBreiteRG * 4 + 1] = y; //Grün-Wert (8bit)
            dataRG[x * 4 + y * imgBreiteRG * 4 + 2] = 0;
            dataRG[x * 4 + y * imgBreiteRG * 4 + 3] = 255;
        }
    }
    osg::ref_ptr<osg::Image> texImgRG = new osg::Image();
    texImgRG->setImage(imgBreiteRG, imgHoeheRG, 1, 4, GL_RGBA, GL_UNSIGNED_BYTE, dataRG, osg::Image::USE_NEW_DELETE, 1);

    //Textur für RG-Werte erstellen und Bild einbinden
    osg::ref_ptr<osg::Texture2D> texturRG = new osg::Texture2D();
    texturRG->setInternalFormat(GL_RGBA);
    texturRG->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
    texturRG->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
    texturRG->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::REPEAT);
    texturRG->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::REPEAT);
    texturRG->setImage(texImgRG.get());

    return texturRG.release();
}

osg::Texture2D *VisScene::makeTexB(void)
{
    // Textur Blau
    //--------------------------------------------------

    //Bild erstellen, in dem Texturkoordinaten pro Pixel gespeichert werden
    //BlauWerte: 1*8Bit
    int imgBreiteB = 16;
    int imgHoeheB = 16;
    unsigned char *dataB;
    dataB = new unsigned char[imgBreiteB * imgHoeheB * 4]; //höhe*breite*farbtiefe[R,G,B,Alpha]
    for (int y = 0; y < imgHoeheB; ++y)
    {
        for (int x = 0; x < imgBreiteB; ++x)
        {
            dataB[x * 4 + y * imgBreiteB * 4 + 0] = 0;
            dataB[x * 4 + y * imgBreiteB * 4 + 1] = 0;
            dataB[x * 4 + y * imgBreiteB * 4 + 2] = x + (y * imgBreiteB); //Blau-Wert (8bit)
            dataB[x * 4 + y * imgBreiteB * 4 + 3] = 255;
        }
    }
    osg::ref_ptr<osg::Image> texImgB = new osg::Image();
    texImgB->setImage(imgBreiteB, imgHoeheB, 1, 4, GL_RGBA, GL_UNSIGNED_BYTE, dataB, osg::Image::USE_NEW_DELETE, 1);

    //Textur für B-Wert erstellen und Bild einbinden
    osg::ref_ptr<osg::Texture2D> texturB = new osg::Texture2D();
    texturB->setInternalFormat(GL_RGBA);
    texturB->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
    texturB->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
    texturB->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP_TO_BORDER);
    texturB->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP_TO_BORDER);
    texturB->setBorderColor(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    texturB->setImage(texImgB.get());

    return texturB.release();
}

osg::TexMat *VisScene::makeTexMatRG()
{
    //Textur-Scaling
    osg::Matrix scaleTex1; //Textur geht von -1 nach 1 wegen vProjMat auf Einheitswürfel, wollen aber von 0 nach 1
    scaleTex1.makeScale(osg::Vec3(0.5f, 0.5f, 1.0f)); //Texturgröße wird halbiert -> Image verdoppelt sich in Größe

    osg::Matrix scaleTex2; //Textur soll 16x in jede Richtung wiederholt werden -> bessere Auflösung
    scaleTex2.makeScale(osg::Vec3(16.0f, 16.0f, 1.0f)); //Scale-Faktor (x,y,z)

    //Textur-Trans
    osg::Matrix transTex1;
    transTex1.makeTranslate(osg::Vec3(0.5f, 0.5f, 0.0f)); //Scale-Faktor (x,y,z)

    //Get Translation from coViewMat
    osg::Matrix transCoView;
    transCoView.makeTranslate(coViewMat.getTrans());

    //Texturmatrix erstellen
    osg::ref_ptr<osg::TexMat> texMatRG = new osg::TexMat();
    texMatRG->setMatrix(/*Scene::getScreen()->getTransMat() 
	                                                * osg::Matrix::rotate(osg::PI/2,osg::X_AXIS) //coViewMat ist in aus VRViewer in GL-KS
							*/ coViewMat
                        * coProjMat
                        * scaleTex1 * transTex1 * scaleTex2);
    return texMatRG.release();
}

osg::TexMat *VisScene::makeTexMatB()
{
    //Textur-Scaling
    osg::Matrix scaleTex1; //Textur geht von -1 nach 1 wegen vProjMat auf Einheitswürfel, wollen aber von 0 nach 1
    scaleTex1.makeScale(osg::Vec3(0.5f, 0.5f, 1.0f)); //Texturgröße wird halbiert -> Image verdoppelt sich in Größe

    //Textur-Trans
    osg::Matrix transTex1;
    transTex1.makeTranslate(osg::Vec3(0.5f, 0.5f, 0.0f)); //Scale-Faktor (x,y,z)

    //Get Translation from coViewMat
    osg::Matrix transCoView;
    transCoView.makeTranslate(coViewMat.getTrans());

    //Texturmatrix erstellen
    osg::ref_ptr<osg::TexMat> texMatB = new osg::TexMat();
    texMatB->setMatrix(/*Scene::getScreen()->getTransMat() 
							*osg::Matrix::rotate(osg::PI/2,osg::X_AXIS) //coViewMat ist in aus VRViewer in GL-KS
							*/ coViewMat
                       * coProjMat
                       * scaleTex1 * transTex1);

    return texMatB.release();
}

osg::Camera *VisScene::makeVisCam()
{
    //----------------------------------------
    //KAMERA zur Darstellung  der Verzerrten Subscene (distQuad im Vollbild)
    //----------------------------------------
    osg::ref_ptr<osg::Camera> visCam = new osg::Camera;

    // just inherit the main cameras view
    visCam->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

    visCam->setViewMatrix(osg::Matrix::rotate(-osg::PI / 2, osg::X_AXIS)); //Kamera schaut in z-Richtung(default), deshalb 90° drehen z->y-Achse
    visCam->setProjectionMatrixAsOrtho2D(-((visResolutionW) / 2.0), //Orthogonalprojektion
                                         ((visResolutionW) / 2.0),
                                         -((visResolutionH) / 2.0),
                                         ((visResolutionH) / 2.0));

    // set the camera to render nach der Projektor-Kamera.
    visCam->setRenderOrder(osg::Camera::POST_RENDER);

    // only clear the depth buffer
    visCam->setClearMask(0);

    return visCam.release();
}
