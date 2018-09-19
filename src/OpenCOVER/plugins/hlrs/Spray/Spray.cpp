﻿/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
**                                                            (C)2018       **
**                                                                          **
** Description: Spray Plugin, qualitiative evaluation of                    **
**              nozzle configurations                                       **
**                                                                          **
** Author: G.Haffner                                                        **
**                                                                          **
** History:                                                                 **
** work is in progress (Sep. 18)                                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "Spray.h"

class nozzleManager* nM = nozzleManager::instance();

parser* parser::_instance = 0;

SprayPlugin::SprayPlugin(): ui::Owner("Spray", cover->ui)
{
    cerr << "Let it spray" << endl;
}


bool SprayPlugin::init()
{
    printf("SprayPlugin::init() start\n");
    parser::instance()->init();
    nM->init();
    idGeo.clear();
    raytracer::instance()->init();


    //Creation of the interface

    sprayMenu_ = new ui::Menu("Spray", this);
    sprayMenu_->setText("Spray");

    globalActions = new ui::Group(sprayMenu_, "Global Actions");

    scaleFactorParticle = new ui::EditField(globalActions, "Particle Scale");
    scaleFactorParticle->setValue(parser::instance()->getScaleFactor());
    scaleFactorParticle->setCallback([this](const std::string &cmd)
    {
        try
        {
            parser::instance()->setScaleFactor(stof(cmd));

        }//try

        catch(const std::invalid_argument& ia)
        {
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
        }
    });

    newGenCreate_ = new ui::EditField(globalActions, "emissionRate");
    newGenCreate_->setText("Emission Rate");
    newGenCreate_->setValue(parser::instance()->getEmissionRate());
    newGenCreate_->setCallback([this](const std::string &cmd)
    {
        try
        {
            parser::instance()->setEmissionRate(stoi(cmd));
            //printf("%i\n", parser::instance()->getNewGenCreate());

        }//try

        catch(const std::invalid_argument& ia)
        {
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
        }
    });

    sprayStart_ = new ui::Button(globalActions, "StartStop");
    sprayStart_->setText("Activate Spray");
    sprayStart_->setCallback([this](bool state)
    {
        if(state == false)
        {
            sprayStart = false;
            std::cout << "Spraying stopped" << std::endl;
        }
        else if(state == true){
            sprayStart = true;
            std::cout << "Spraying started" << std::endl;
        }

    });

    loadSaveMenu_ = new ui::Menu(globalActions, "Save/Load Config");

    pathNameFielddyn_ = new ui::EditField(loadSaveMenu_, "pathname");
    pathNameFielddyn_->setText("Path Name");
    pathNameFielddyn_->setCallback([this](const std::string &cmd)
    {
        pathNameField_ = cmd;
    });

    fileNameFielddyn_ = new ui::EditField(loadSaveMenu_, "");
    fileNameFielddyn_->setText("File Name");
    fileNameFielddyn_->setCallback([this](const std::string &cmd)
    {
        fileNameField_ = cmd;
    });

    ui::Action* acceptL = new ui::Action(loadSaveMenu_, "Load nozzles");
    acceptL->setCallback([this]()
    {
        nM->loadNozzle(pathNameField_, fileNameField_);
        while(nM->checkAll() != NULL)
        {
            nozzle* temporary = nM->checkAll();
            temporary->registerLabel();

            std::stringstream ss;
            ss << temporary->getName();
            nozzleIDL->append(ss.str());
        }
        if(nM->getNozzle(nozzleID))
        {
            edit_->setEnabled(true);
            remove_->setEnabled(true);
        }
    });

    ui::Action* acceptS = new ui::Action(loadSaveMenu_, "Save nozzles");
    acceptS->setCallback([this]()
    {
        nM->saveNozzle(pathNameField_, fileNameField_);
    });




    bbEditMenu = new ui::Menu(globalActions, "BoundingBox Editor");

    ui::EditField* xBB = new ui::EditField(bbEditMenu, "BoundingBox X");
    xBB->setValue(nM->getBoundingBox().x());
    xBB->setCallback([this](const std::string &cmd)
    {
        try
        {
            osg::Vec3 tempBoundingBox = nM->getBoundingBox();
            tempBoundingBox.x() = stof(cmd);
            nM->setBoundingBox(tempBoundingBox);

        }//try

        catch(const std::invalid_argument& ia)
        {
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
        }
    });

    ui::EditField* yBB = new ui::EditField(bbEditMenu, "BoundingBox Y");
    yBB->setValue(nM->getBoundingBox().x());
    yBB->setCallback([this](const std::string &cmd)
    {
        try
        {
            osg::Vec3 tempBoundingBox = nM->getBoundingBox();
            tempBoundingBox.y() = stof(cmd);
            nM->setBoundingBox(tempBoundingBox);

        }//try

        catch(const std::invalid_argument& ia)
        {
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
        }
    });

    ui::EditField* zBB = new ui::EditField(bbEditMenu, "BoundingBox Z");
    zBB->setValue(nM->getBoundingBox().x());
    zBB->setCallback([this](const std::string &cmd)
    {
        try
        {
            osg::Vec3 tempBoundingBox = nM->getBoundingBox();
            tempBoundingBox.z() = stof(cmd);
            nM->setBoundingBox(tempBoundingBox);

        }//try

        catch(const std::invalid_argument& ia)
        {
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
        }
    });

    ui::Action* resetScene = new ui::Action(globalActions, "resetScene");
    resetScene->setText("Reset RT Scene");
    resetScene->setCallback([this]()
    {
        raytracer::instance()->removeAllGeometry();                     //resets scene
        nodeVisitorVertex c;                                            //creates new scene
        cover->getObjectsRoot()->accept(c);
        raytracer::instance()->createFaceSet(c.getVertexArray(),0);
        raytracer::instance()->finishAddGeometry();
    });

    autoremove = new ui::Button(globalActions, "autoremove");
    autoremove->setText("Autoremove particles");
    autoremove->setCallback([this](bool state)
    {
        if(state == false)
        {
            nM->autoremove(false);
            std::cout << "Autoremove stopped" << std::endl;
        }
        else
            if(state == true){
                nM->autoremove(true);
                std::cout << "Autoremove started" << std::endl;
            }
    });

    ui::EditField* numParticleField = new ui::EditField(globalActions, "Num of Particles");
    numParticleField->setValue(parser::instance()->getReqParticles());
    numParticleField->setCallback([this](std::string cmd)
    {
        try
        {
            int numParticles = stof(cmd);
            parser::instance()->setNumParticles(numParticles);
            raytracer::instance()->setNumRays(numParticles);

        }//try

        catch(const std::invalid_argument& ia)
        {
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
            newColor.x() = 1;
        }//catch
    });

    nozzleActions = new ui::Group(sprayMenu_, "Nozzle Actions");

    nozzleIDL = new ui::SelectionList(nozzleActions, "nozzleSelection");
    nozzleIDL->setText("Selected nozzle");
    nozzleIDL->setCallback([this](int val){
        editNozzle = nM->getNozzle(val);
        nozzleID = val;
        if(editNozzle == nullptr)
        {
            remove_->setEnabled(false);
            edit_->setEnabled(false);
            if(nozzleEditMenu_ != nullptr)
                nozzleEditMenu_->setEnabled(false);
        }
        else
        {
            remove_->setEnabled(true);
            edit_->setEnabled(true);
            if(nozzleEditMenu_ != nullptr)
            {
                if(edit_->state())
                {
                    nozzleEditMenu_->setVisible(true);
                    nozzleEditMenu_->setEnabled(true);
                }
                updateEditContext();
            }
        }
    });

    edit_ = new ui::Button(nozzleActions, "editContext");
    edit_->setText("Open Edit Menu");
    edit_->setEnabled(false);
    edit_->setCallback([this](bool state)
    {
        std::cout << "Editing nozzle ID: " << nozzleID << " started" << std::endl;
        if(editNozzle == nullptr)
            editNozzle = nM->getNozzle(nozzleID);

        if(nullptr != editNozzle && state)
        {
            if(parser::instance()->getSphereRenderType() == 0)
                editNozzle->setColor(osg::Vec4(1,1,0,1));
            newColor = editNozzle->getColor();

            if(nozzleEditMenu_ != nullptr)
            {
                nozzleEditMenu_->setEnabled(true);
                nozzleEditMenu_->setVisible(true);
                updateEditContext();
            }
            else
            {
                nozzleEditMenu_ = new ui::Menu(nozzleActions, "EditingInterface");
                nozzleEditMenu_->setText("Editing Interface");

                //Set variables of nozzle to init values
                minimum = editNozzle->getMinimum()*1000000;
                deviation = editNozzle->getDeviation()*1000000;

                red_ = new ui::EditField(nozzleEditMenu_, "redField");
                red_->setText("Red value");
                red_->setValue(editNozzle->getColor().x());
                red_->setCallback([this](const std::string &cmd)
                {
                    try
                    {
                        newColor.x() = stof(cmd);

                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                        newColor.x() = 1;
                    }//catch
                });

                green_ = new ui::EditField(nozzleEditMenu_, "greenField");
                green_->setText("Green value");
                green_->setValue(editNozzle->getColor().y());
                green_->setCallback([this](const std::string &cmd)
                {
                    try
                    {
                        newColor.y() = stof(cmd);

                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                        newColor.y() = 1;
                    }//catch
                });

                blue_ = new ui::EditField(nozzleEditMenu_, "blueField");
                blue_->setText("Blue value");
                blue_->setValue(editNozzle->getColor().z());
                blue_->setCallback([this](const std::string &cmd)
                {
                    try
                    {
                        newColor.z() = stof(cmd);

                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                        newColor.z() = 1;
                    }//catch
                });

                alpha_ = new ui::EditField(nozzleEditMenu_, "alphaField");
                alpha_->setText("Alpha value");
                alpha_->setValue(editNozzle->getColor().w());
                alpha_->setCallback([this](const std::string &cmd)
                {
                    try
                    {
                        newColor.w() = stof(cmd);

                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                        newColor.w() = 1;
                    }//catch
                });

                pressureSlider_ = new ui::Slider(nozzleEditMenu_, "sliderPressure");
                pressureSlider_->setText("Initial pressure in bar");
                pressureSlider_->setBounds(parser::instance()->getLowerPressureBound(),
                                           parser::instance()->getUpperPressureBound()
                                           );
                pressureSlider_->setValue(editNozzle->getInitPressure());
                pressureSlider_->setCallback([this](float value, bool state)
                {
                    editNozzle->setInitPressure(value);
                });

                alphaSlider_ = new ui::Slider(nozzleEditMenu_, "sliderAlpha");
                alphaSlider_->setText("Gaussian Alpha Value");
                alphaSlider_->setBounds(0.1, 1);
                alphaSlider_->setValue(editNozzle->getAlpha());
                alphaSlider_->setCallback([this](float value, bool state)
                {
                    editNozzle->setAlpha(value);
                });


                sizeSlider_ = new ui::Slider(nozzleEditMenu_, "sliderSize");
                sizeSlider_->setText("Size of Nozzle");
                sizeSlider_->setBounds(0.1, 100);
                sizeSlider_->setValue(editNozzle->getScale());
                sizeSlider_->setCallback([this](float value, bool state)
                {
                    editNozzle->setScale(value);
                });

                rMinimum = new ui::EditField(nozzleEditMenu_, "Minimum in microns");
                rMinimum->setValue(editNozzle->getMinimum()*1000000);
                rMinimum->setCallback([this](const std::string &cmd)
                {
                    try
                    {
                        minimum = stof(cmd);

                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                    }//catch
                });

                rDeviation = new ui::EditField(nozzleEditMenu_, "Deviation in microns");
                rDeviation->setValue(editNozzle->getDeviation()*1000000);
                rDeviation->setCallback([this](const std::string &cmd){
                    manager()->update();

                    try
                    {
                        deviation = stof(cmd);

                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                    }//catch
                });

                param1 = new ui::EditField(nozzleEditMenu_, "param1");
                if(editNozzle->getType().compare("standard") == 0)
                    param1->setText("Spray Angle");
                if(editNozzle->getType().compare("image") == 0)
                    param1->setText("Path Name");
                param1->setValue(editNozzle->getParam1());

                param2 = new ui::EditField(nozzleEditMenu_, "param2");
                if(editNozzle->getType().compare("standard") == 0)
                    param2->setText("Decoy");
                if(editNozzle->getType().compare("image") == 0)
                    param2->setText("File Name");
                param2->setValue(editNozzle->getParam2());

                interaction = new ui::Button(nozzleEditMenu_, "InteractionNozzle");
                interaction->setText("Interaction");
                interaction->setState(editNozzle->getIntersection());
                interaction->setCallback([this](bool state){
                    if(state == false)
                    {
                        editNozzle->disableIntersection();
                        editNozzle->setIntersection(false);
                        editNozzle->display(false);
                        std::cout << "Interaction deactivated" << std::endl;
                    }
                    else
                        if(state == true){
                            editNozzle->enableIntersection();
                            editNozzle->setIntersection(true);
                            editNozzle->display(true);
                            std::cout << "Interaction activated" << std::endl;
                        }

                });

                acceptEdit_ = new ui::Action(nozzleEditMenu_, "acceptEdit");
                acceptEdit_->setText("Accept");
                acceptEdit_->setCallback([this](){
                    if(editNozzle != nullptr)
                    {
                        if(parser::instance()->getSphereRenderType() == 0)
                            editNozzle->setColor(newColor);   //Somehow crashes the rendering of spheres
                        editNozzle->setInitPressure(pressureSlider_->value());
                        editNozzle->setAlpha(alphaSlider_->value());
                        editNozzle->setMinimum(minimum/1000000);
                        editNozzle->setDeviation(deviation/1000000);
                        std::cout << "Editing done" << std::endl;
                    }
                    //nozzleEditMenu_->setVisible(false);

                });

                controller = new ui::Button(nozzleEditMenu_, "Open Controller");
                controller->setCallback([this](bool state)
                {
                    if(state)
                        testMenu->setVisible(true);
                    else
                        testMenu->setVisible(false);
                });

                testMenu = new ui::Menu(nozzleEditMenu_, "tester");
                testMenu->setText("Controller");

                memMat.makeIdentity();

                rotX = new ui::Slider(testMenu, "rotX");
                rotX->setText("Rotation X-Axis");
                rotX->setBounds(-1,1);
                rotX->setValue(editNozzle->getMatrix().getRotate().x());
                rotX->setCallback([this](float value, bool stop){
                    memMat = editNozzle->getMatrix();
                    osg::Quat a = editNozzle->getMatrix().getRotate();
                    a.x() = value;
                    memMat.setRotate(a);
                    editNozzle->updateTransform(memMat);

                });

                rotY = new ui::Slider(testMenu, "rotY");
                rotY->setText("Rotation Y-Axis");
                rotY->setBounds(-1,1);
                rotY->setValue(editNozzle->getMatrix().getRotate().y());
                rotY->setCallback([this](float value, bool stop){
                    memMat = editNozzle->getMatrix();
                    osg::Quat a = editNozzle->getMatrix().getRotate();
                    a.y() = value;
                    memMat.setRotate(a);
                    editNozzle->updateTransform(memMat);

                });

                rotZ = new ui::Slider(testMenu, "rotZ");
                rotZ->setText("Rotation Z-Axis");
                rotZ->setBounds(-1,1);
                rotZ->setValue(editNozzle->getMatrix().getRotate().z());
                rotZ->setCallback([this](float value, bool stop){
                    memMat = editNozzle->getMatrix();
                    osg::Quat a = editNozzle->getMatrix().getRotate();
                    a.z() = value;
                    memMat.setRotate(a);
                    editNozzle->updateTransform(memMat);

                });


                moveX = new ui::EditField(testMenu, "moveXfield");
                moveX->setText("Move X");
                moveX->setValue(editNozzle->getMatrix().getTrans().x());
                moveX->setCallback([this](const std::string &cmd){

                    try
                    {
                        transMat.x() = stof(cmd);
                        memMat = editNozzle->getMatrix();
                        memMat.setTrans(transMat);
                        editNozzle->updateTransform(memMat);
                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                        transMat.x() = 0;
                    }
                });

                moveY = new ui::EditField(testMenu, "moveYfield");
                moveY->setText("Move Y");
                moveY->setValue(editNozzle->getMatrix().getTrans().y());
                moveY->setCallback([this](const std::string &cmd)
                {
                    try
                    {
                        transMat.y() = stof(cmd);
                        memMat = editNozzle->getMatrix();
                        memMat.setTrans(transMat);;
                        editNozzle->updateTransform(memMat);

                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                        transMat.y() = 0;
                    }
                });

                moveZ = new ui::EditField(testMenu, "moveZfield");
                moveZ->setText("Move Z");
                moveZ->setValue(editNozzle->getMatrix().getTrans().z());
                moveZ->setCallback([this](const std::string &cmd)
                {
                    try
                    {
                        transMat.z() = stof(cmd);
                        memMat = editNozzle->getMatrix();
                        memMat.setTrans(transMat);
                        editNozzle->updateTransform(memMat);

                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                        transMat.z() = 0;
                    }
                });

                ui::Action* setToCurPos = new ui::Action(testMenu, "setToCurPos");
                setToCurPos->setText("Set to current position");
                setToCurPos->setCallback([this]()
                {
                    osg::Matrix newPos = editNozzle->getMatrix();
                    newPos.setTrans(cover->getMouseMat().getTrans());
                    editNozzle->updateTransform(newPos);
                });

                testMenu->setVisible(false);
            }
            //#endif
        }
        else
            nozzleEditMenu_->setVisible(false);
    });

    nozzleCreateMenuImage = new ui::Menu(sprayMenu_, "Image Nozzle Parameters");
    ui::EditField* subMenuPathname_ = new ui::EditField(nozzleCreateMenuImage, "pathname_");
    subMenuPathname_->setText("Path Name");
    subMenuPathname_->setCallback([this](const std::string &cmd)
    {
        pathNameField_ = cmd;
    });
    ui::EditField* subMenuFilename_ = new ui::EditField(nozzleCreateMenuImage, "filename_");
    subMenuFilename_->setText("File Name");
    subMenuFilename_->setCallback([this](const std::string &cmd)
    {
        fileNameField_ = cmd;
    });
    ui::EditField* subMenuNozzlenameI_ = new ui::EditField(nozzleCreateMenuImage, "nozzlename_");
    subMenuNozzlenameI_->setText("Nozzle Name");
    subMenuNozzlenameI_->setCallback([this](const std::string &cmd)
    {
        nozzleNameField_ = cmd;
    });

    ui::Action* acceptI = new ui::Action(nozzleCreateMenuImage, "acceptImage");
    acceptI->setText("Accept");
    acceptI->setCallback([this]()
    {
        class nozzle* temporary = nM->createImageNozzle(nozzleNameField_.c_str(),
                                                        pathNameField_,
                                                        fileNameField_);
        if(temporary != NULL)
        {
            temporary->registerLabel();

            nozzleIDL->append(temporary->getName());

            if(nM->getNozzle(nozzleIDL->selectedIndex()) != nullptr|| nM->getNozzle(nozzleID) != nullptr)
            {
                edit_->setEnabled(true);
                remove_->setEnabled(true);
            }
        }
    });

    nozzleCreateMenuStandard = new ui::Menu(sprayMenu_, "Standard Nozzle Parameter");
    ui::EditField* subMenuSprayAngle_ = new ui::EditField(nozzleCreateMenuStandard, "sprayAngle_");
    subMenuSprayAngle_->setText("Spray Angle");
    subMenuSprayAngle_->setCallback([this](const std::string &cmd)
    {
        try
        {
            sprayAngle_ = stof(cmd);

        }//try

        catch(const std::invalid_argument& ia)
        {
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
            sprayAngle_ = 0;
        }//catch

    });
    ui::EditField* subMenuDecoy_ = new ui::EditField(nozzleCreateMenuStandard, "decoy_");
    subMenuDecoy_->setText("Decoy");
    subMenuDecoy_->setCallback([this](const std::string &cmd)
    {
        decoy_ = cmd.c_str();
    });
    ui::EditField* subMenuNozzlenameS_ = new ui::EditField(nozzleCreateMenuStandard, "nozzlename_");
    subMenuNozzlenameS_->setText("Nozzle Name");
    subMenuNozzlenameS_->setCallback([this](const std::string &cmd)
    {
        nozzleNameField_ = cmd;
    });
    ui::Action* acceptStandard = new ui::Action(nozzleCreateMenuStandard, "acceptStandard");
    acceptStandard->setText("Accept");
    acceptStandard->setCallback([this]()
    {
        class nozzle* temporary = nM->createStandardNozzle(nozzleNameField_.c_str(),
                                                           sprayAngle_,
                                                           decoy_);
        temporary->registerLabel();

        nozzleIDL->append(temporary->getName());

        if(nM->getNozzle(nozzleIDL->selectedIndex()) != nullptr|| nM->getNozzle(nozzleID) != nullptr)
        {
            edit_->setEnabled(true);
            remove_->setEnabled(true);
        }


    });

    remove_ = new ui::Action(nozzleActions, "removeNozzle");
    remove_->setText("Remove Nozzle");
    remove_->setEnabled(false);
    remove_->setCallback([this]()
    {
        if(nM->getNozzle(nozzleID) != 0)
        {
            nozzleIDL->select(nozzleID, false);

            auto selList = nozzleIDL->items();
            selList[nozzleID] = "deleted";
            nozzleIDL->setList(selList);;
            editNozzle = nM->getNozzle(nozzleID);
            nM->removeNozzle(nozzleID);
            editNozzle = nullptr;
            edit_->setEnabled(false);
            remove_->setEnabled(false);
            nozzleEditMenu_->setEnabled(false);
        }
        else
        {
            std::cout << "The nozzle doesn't exist or is already deleted" << std::endl;
        }
    });


    scene = new osg::Group;
    cover->getObjectsRoot()->addChild(scene);
    scene->setName("Spray Group");
    testBoxGeode = new osg::Geode;
    testBoxGeode->setName("testBox");

    //Just for testing purpose
    createTestBox(osg::Vec3(0,0,-10), osg::Vec3(10,10,10));
    //createTestBox1(osg::Vec3(0,20,-20), osg::Vec3(10,10,10), true);

    scene->addChild(testBoxGeode);

    //Traverse scenegraph to extract vertices for raytracer
    nodeVisitorVertex c;

    std::clock_t begin = clock();

    cover->getObjectsRoot()->accept(c);

    std::clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    printf("elapsed time for traversing %f, vertices read out %i\n", elapsed_secs, c.numOfVertices);

    //Set nozzle geometry for VRML nozzles
    for(int i = 0; i < c.coNozzleList.size(); i++)
    {
        class nozzle* current = nM->createStandardNozzle("", 30, "NONE");
        current->setNozzleGeometryNode(c.coNozzleList[i]);
        current->registerLabel();
        nozzleIDL->append(current->getName());
        edit_->setEnabled(true);
        remove_->setEnabled(true);
    }

    begin = clock();

    raytracer::instance()->createFaceSet(c.getVertexArray(),0);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    printf("elapsed time for generating in embree %f\n", elapsed_secs);


    raytracer::instance()->finishAddGeometry();

    printf("SprayPlugin::init() finished\n");

    return true;

}

bool SprayPlugin::destroy()
{
    scene->removeChild(floorGeode);
    scene->removeChild(testBoxGeode);
    cover->getObjectsRoot()->removeChild(scene);
    cover->getObjectsRoot()->removeChild(testBoxGeode);
    nM->remove_all();

    delete sprayStart_;
    delete save_;
    delete load_;
    delete create_;
    delete remove_;;
    delete pathNameFielddyn_;
    delete fileNameFielddyn_;
    delete nozzleNameFielddyn_;
    delete sprayMenu_;

    return true;
}

bool SprayPlugin::update()
{
    if(sprayStart == true)
        nM->update();

    return true;
}

void SprayPlugin::createTestBox(osg::Vec3 initPos, osg::Vec3 scale)
{
    osg::Box* testBox = new osg::Box(osg::Vec3(initPos.x(), initPos.z(), initPos.y()), 10);
    osg::TessellationHints *hints = new osg::TessellationHints();
    hints->setDetailRatio(0.5);
    osg::ShapeDrawable *boxDrawableTest = new osg::ShapeDrawable(testBox, hints);
    boxDrawableTest->setColor(osg::Vec4(0, 0.5, 0, 1));
    testBoxGeode->addDrawable(boxDrawableTest);

    idGeo.push_back(raytracer::instance()->createCube(initPos, scale));
}

void SprayPlugin::createTestBox1(osg::Vec3 initPos, osg::Vec3 scale, bool manual)
{
    osg::Geometry *geom = new osg::Geometry;

    osg::Vec3Array *vertices = new osg::Vec3Array;
    vertices->push_back(osg::Vec3(-1*scale.x(), -1*scale.y(), -1*scale.z()) +initPos);
    vertices->push_back(osg::Vec3(-1*scale.x(), -1*scale.y(), 1*scale.z())  +initPos);
    vertices->push_back(osg::Vec3(-1*scale.x(), 1*scale.y(), -1*scale.z())  +initPos);
    vertices->push_back(osg::Vec3(-1*scale.x(), 1*scale.y(), 1*scale.z())   +initPos);
    vertices->push_back(osg::Vec3(1*scale.x(), -1*scale.y(), -1*scale.z())  +initPos);
    vertices->push_back(osg::Vec3(1*scale.x(), -1*scale.y(), 1*scale.z())   +initPos);
    vertices->push_back(osg::Vec3(1*scale.x(), 1*scale.y(), -1*scale.z())   +initPos);
    vertices->push_back(osg::Vec3(1*scale.x(), 1*scale.y(), 1*scale.z())    +initPos);

    geom->setVertexArray(vertices);

    osg::Vec4Array *colors = new osg::Vec4Array;
    colors->push_back(osg::Vec4(1,0,0,1));
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::Vec3Array *normals = new osg::Vec3Array;
    normals->push_back(osg::Vec3(0,0,-1));
    geom->setNormalArray(normals);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,8));

    testBoxGeode->addDrawable(geom);

}

void SprayPlugin::updateEditContext()
{
    //nozzleEditMenu_->setVisible(true);
    red_->setValue(editNozzle->getColor().x());
    green_->setValue(editNozzle->getColor().x());
    blue_->setValue(editNozzle->getColor().x());
    alpha_->setValue(editNozzle->getColor().x());
    pressureSlider_->setValue(editNozzle->getInitPressure());
    alphaSlider_->setValue(editNozzle->getAlpha());
    sizeSlider_->setValue(editNozzle->getScale());
    rotX->setValue(editNozzle->getMatrix().getRotate().x());
    rotY->setValue(editNozzle->getMatrix().getRotate().y());
    rotZ->setValue(editNozzle->getMatrix().getRotate().z());
    moveX->setValue(editNozzle->getMatrix().getTrans().x());
    moveY->setValue(editNozzle->getMatrix().getTrans().y());
    moveZ->setValue(editNozzle->getMatrix().getTrans().z());
    minimum = editNozzle->getMinimum()*1000000;
    rMinimum->setValue(editNozzle->getMinimum()*1000000);
    deviation = editNozzle->getDeviation()*1000000;
    rDeviation->setValue(editNozzle->getDeviation()*1000000);

    if(editNozzle->getType().compare("standard") == 0)
        param1->setText("Spray Angle");
    if(editNozzle->getType().compare("image") == 0)
        param1->setText("Path Name");
    param1->setValue(editNozzle->getParam1());

    if(editNozzle->getType().compare("standard") == 0)
        param2->setText("Decoy");
    if(editNozzle->getType().compare("image") == 0)
        param2->setText("File Name");
    param2->setValue(editNozzle->getParam2());

    interaction->setState(editNozzle->getIntersection());
}


COVERPLUGIN(SprayPlugin)
