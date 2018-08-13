/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//source .covise.sh set environment variables

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

    nozzleIDL = new ui::SelectionList(sprayMenu_, "nozzleSelection");
    nozzleIDL->setText("Selected nozzle");
    nozzleIDL->setCallback([this](int val){
        editNozzle = nM->getNozzle(val);
        nozzleID = val;
    });

    edit_ = new ui::Action(sprayMenu_, "editContext");
    edit_->setText("Edit Nozzle");
    edit_->setCallback([this]()
    {
        if(editing == false)
        {
            editing = true;
            currentNozzleID = nozzleID;
            std::cout << "Editing nozzle ID: " << currentNozzleID << " started" << std::endl;

            if(0 != nM->getNozzle(currentNozzleID))
            {
                editNozzle = nM->getNozzle(currentNozzleID);
                //editNozzle->setColor(osg::Vec4(1,1,0,1));
                newColor = editNozzle->getColor();

                if(nozzleEditMenu_ != nullptr)
                {
                    nozzleEditMenu_->setVisible(true);
                    red_->setValue(editNozzle->getColor().x());
                    green_->setValue(editNozzle->getColor().x());
                    blue_->setValue(editNozzle->getColor().x());
                    alpha_->setValue(editNozzle->getColor().x());
                    pressureSlider_->setValue(editNozzle->getInitPressure());
                    rotX->setValue(editNozzle->getMatrix().getRotate().x());
                    rotY->setValue(editNozzle->getMatrix().getRotate().y());
                    rotZ->setValue(editNozzle->getMatrix().getRotate().z());
                    moveX->setValue(editNozzle->getMatrix().getTrans().x());
                    moveY->setValue(editNozzle->getMatrix().getTrans().y());
                    moveZ->setValue(editNozzle->getMatrix().getTrans().z());

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

                }
                else
                {
                    nozzleEditMenu_ = new ui::Menu("Spray", this);
                    nozzleEditMenu_->setText("Spray");

                red_ = new ui::EditField(nozzleEditMenu_, "redField");
                red_->setText("Red value");
                red_->setValue(editNozzle->getColor().x());
                red_->setCallback([this](const std::string &cmd){
                    manager()->update();

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
                green_->setCallback([this](const std::string &cmd){
                    manager()->update();

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
                blue_->setCallback([this](const std::string &cmd){
                    manager()->update();

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
                alpha_->setCallback([this](const std::string &cmd){
                    manager()->update();

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
                pressureSlider_->setText("Initial pressure");
                pressureSlider_->setBounds(parser::instance()->getLowerPressureBound(),
                                           parser::instance()->getUpperPressureBound()
                                           );
                pressureSlider_->setValue(editNozzle->getInitPressure());

                ui::EditField* rMinimum = new ui::EditField(nozzleEditMenu_, "Minimum");
                rMinimum->setValue(editNozzle->getMinimum());
                rMinimum->setCallback([this](const std::string &cmd){
                    manager()->update();

                    try
                    {
                        minimum = stof(cmd);

                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                    }//catch
                });

                ui::EditField* rDeviation = new ui::EditField(nozzleEditMenu_, "Deviation");
                rDeviation->setValue(editNozzle->getDeviation());
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

                acceptEdit_ = new ui::Action(nozzleEditMenu_, "acceptEdit");
                acceptEdit_->setText("Accept");
                acceptEdit_->setCallback([this](){
                    //editNozzle->setColor(newColor);                               //Somehow crashes the rendering of spheres
                    editNozzle->setInitPressure(pressureSlider_->value());
                    editNozzle->setMinimum(minimum);
                    editNozzle->setDeviation(deviation);
                    editing = false;
                    std::cout << "Editing done" << std::endl;
                    nozzleEditMenu_->setVisible(false);

                });

//#if TESTING
                testMenu = new ui::Menu(nozzleEditMenu_, "tester");
                testMenu->setText("Controller");

                memMat.makeIdentity();

                rotX = new ui::Slider(testMenu, "rotX");
                rotX->setText("Rotation X-Axis");
                rotX->setBounds(-1,1);
                rotX->setValue(editNozzle->getMatrix().getRotate().x());
                rotX->setCallback([this](float value, bool stop){
                    osg::Vec3 trans = editNozzle->getMatrix().getTrans();
                    osg::Quat a = editNozzle->getMatrix().getRotate();
                    a.x() = value;
                    memMat.makeIdentity();
                    memMat.setRotate(a);
                    memMat.setTrans(trans);
                    editNozzle->updateTransform(memMat);

                });

                rotY = new ui::Slider(testMenu, "rotY");
                rotY->setText("Rotation Y-Axis");
                rotY->setBounds(-1,1);
                rotY->setValue(editNozzle->getMatrix().getRotate().y());
                rotY->setCallback([this](float value, bool stop){
                    osg::Vec3 trans = editNozzle->getMatrix().getTrans();
                    osg::Quat a = editNozzle->getMatrix().getRotate();
                    a.y() = value;;
                    memMat.makeIdentity();
                    memMat.setRotate(a);
                    memMat.setTrans(trans);
                    editNozzle->updateTransform(memMat);

                });

                rotZ = new ui::Slider(testMenu, "rotZ");
                rotZ->setText("Rotation Z-Axis");
                rotZ->setBounds(-1,1);
                rotZ->setValue(editNozzle->getMatrix().getRotate().z());
                rotZ->setCallback([this](float value, bool stop){
                    osg::Vec3 trans = editNozzle->getMatrix().getTrans();
                    osg::Quat a = editNozzle->getMatrix().getRotate();
                    a.z() = value;
                    memMat.makeIdentity();
                    memMat.setRotate(a);
                    memMat.setTrans(trans);
                    editNozzle->updateTransform(memMat);

                });


                moveX = new ui::EditField(testMenu, "moveXfield");
                moveX->setText("Move X");
                moveX->setValue(editNozzle->getMatrix().getTrans().x());
                moveX->setCallback([this](const std::string &cmd){
                    //manager()->update();

                    try
                    {
                        transMat.x() = stof(cmd);
                        memMat = editNozzle->getMatrix();
                        memMat.setTrans(transMat);
                        editNozzle->updateTransform(memMat);
                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        outputField_->setText("Value of Move X must be an integer");
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                        transMat.x() = 0;
                    }
                });

                moveY = new ui::EditField(testMenu, "moveYfield");
                moveY->setText("Move Y");
                moveY->setValue(editNozzle->getMatrix().getTrans().y());
                moveY->setCallback([this](const std::string &cmd){
                    manager()->update();

                    try
                    {
                        transMat.y() = stof(cmd);
                        memMat = editNozzle->getMatrix();
                        memMat.setTrans(transMat);;
                        editNozzle->updateTransform(memMat);

                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        outputField_->setText("Value of Move Y must be an integer");
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                        transMat.y() = 0;
                    }
                });

                moveZ = new ui::EditField(testMenu, "moveZfield");
                moveZ->setText("Move Z");
                moveZ->setValue(editNozzle->getMatrix().getTrans().z());
                moveZ->setCallback([this](const std::string &cmd){
                    manager()->update();

                    try
                    {
                        transMat.z() = stof(cmd);
                        memMat = editNozzle->getMatrix();
                        memMat.setTrans(transMat);
                        editNozzle->updateTransform(memMat);

                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        outputField_->setText("Value of Move Z must be an integer");
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                        transMat.z() = 0;
                    }
                });

                ui::Action* setToCurPos = new ui::Action(testMenu, "setToCurPos");
                setToCurPos->setText("Set to current position");
                setToCurPos->setCallback([this](){
                    osg::Matrix newPos = editNozzle->getMatrix();
                    newPos.setTrans(cover->getViewerMat().getTrans());
                    editNozzle->updateTransform(newPos);
                });
            }
//#endif
            }
        }
        else editing = false;
    });

    sprayStart_ = new ui::Button(sprayMenu_, "StartStop");
    sprayStart_->setText("Activate Spray");
    sprayStart_->setCallback([this](bool state){
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

    save_ = new ui::Action(sprayMenu_, "Save");
    save_->setText("Save");
    save_->setCallback([this](){
        manager()->update();
        if(saveLoadMenu_ != nullptr)
        {
            saveLoadMenu_->setVisible(true);
        }
        else
        {
            saveLoadMenu_ = new ui::Menu(sprayMenu_, "Save or Load Nozzles");

            pathNameFielddyn_ = new ui::EditField(saveLoadMenu_, "pathname");
            pathNameFielddyn_->setText("Path Name");
            pathNameFielddyn_->setCallback([this](const std::string &cmd){
                manager()->update();
                pathNameField_ = cmd;
            });

            fileNameFielddyn_ = new ui::EditField(saveLoadMenu_, "");
            fileNameFielddyn_->setText("File Name");
            fileNameFielddyn_->setCallback([this](const std::string &cmd){
                manager()->update();
                fileNameField_ = cmd;
            });

            ui::Action* acceptSL = new ui::Action(saveLoadMenu_, "Accept");
            acceptSL->setCallback([this](){
                nM->saveNozzle(pathNameField_, fileNameField_);
                saveLoadMenu_->setVisible(false);
            });
        }


    });

    load_ = new ui::Action(sprayMenu_, "Load");
    load_->setText("Load");
    load_->setCallback([this](){
        if(saveLoadMenu_ != nullptr)
        {
            saveLoadMenu_->setVisible(true);
        }
        else
        {
            saveLoadMenu_ = new ui::Menu(sprayMenu_, "Save or Load Nozzles");

            pathNameFielddyn_ = new ui::EditField(sprayMenu_, "pathname");
            pathNameFielddyn_->setText("Path Name");
            pathNameFielddyn_->setCallback([this](const std::string &cmd){
                manager()->update();
                pathNameField_ = cmd;
            });

            fileNameFielddyn_ = new ui::EditField(sprayMenu_, "");
            fileNameFielddyn_->setText("File Name");
            fileNameFielddyn_->setCallback([this](const std::string &cmd){
                manager()->update();
                fileNameField_ = cmd;
            });

            ui::Action* acceptSL = new ui::Action(saveLoadMenu_, "Accept");
            acceptSL->setCallback([this](){
                nM->loadNozzle(pathNameField_.c_str(), fileNameField_.c_str());
                while(nM->checkAll() != NULL)
                {
                    nozzle* temporary = nM->checkAll();
                    temporary->registerLabel();

                    std::stringstream ss;
                    ss << temporary->getName() << " " << temporary->getID();
                    nozzleIDL->append(ss.str());
                }
                saveLoadMenu_->setVisible(false);
            });
        }

    });

    create_ = new ui::Action(sprayMenu_, "createNozzle");
    create_->setText("Create Nozzle");
    create_->setCallback([this](){

        if(creating == false)
        {
            creating = true;

            if(nozzleCreateMenu != nullptr)
                nozzleCreateMenu->setVisible(true);
            else
            {
            nozzleCreateMenu = new ui::Menu(sprayMenu_, "Type of nozzle");
            outputField_->setText("Choose your type of nozzle");

            ui::Action* createImage = new ui::Action(nozzleCreateMenu, "image");
            createImage->setText("Create image nozzle");
            createImage->setCallback([this](){
                if(nozzleCreateMenuImage != nullptr)
                    nozzleCreateMenuImage->setVisible(true);
                else
                {
                    nozzleCreateMenuImage = new ui::Menu(nozzleCreateMenu, "Image Nozzle Parameters");
                    outputField_->setText("Set the parameters of the image nozzle");
                    ui::EditField* subMenuPathname_ = new ui::EditField(nozzleCreateMenuImage, "pathname_");
                    subMenuPathname_->setText("Path Name");
                    subMenuPathname_->setCallback([this](const std::string &cmd){
                        manager()->update();
                        pathNameField_ = cmd;
                    });
                    ui::EditField* subMenuFilename_ = new ui::EditField(nozzleCreateMenuImage, "filename_");
                    subMenuFilename_->setText("File Name");
                    subMenuFilename_->setCallback([this](const std::string &cmd){
                        manager()->update();
                        fileNameField_ = cmd;
                    });
                    ui::EditField* subMenuNozzlename_ = new ui::EditField(nozzleCreateMenuImage, "nozzlename_");
                    subMenuNozzlename_->setText("Nozzle Name");
                    subMenuNozzlename_->setCallback([this](const std::string &cmd){
                        manager()->update();
                        nozzleNameField_ = cmd;
                    });

                    ui::Action* accept = new ui::Action(nozzleCreateMenuImage, "acceptImage");
                    accept->setText("Accept");
                    accept->setCallback([this](){
                        createAndRegisterImageNozzle();
                        creating = false;
                        //delete tempMenu;
                        nozzleCreateMenuImage->setVisible(false);
                        nozzleCreateMenu->setVisible(false);
                        outputField_->setText("Help Field");
                    });
                }

            });

            ui::Action* createStandard = new ui::Action(nozzleCreateMenu, "standard");
            createStandard->setText("Create standard nozzle");
            createStandard->setCallback([this](){
                if(nozzleCreateMenuStandard != nullptr)
                    nozzleCreateMenuStandard->setVisible(true);
                else
                {
                    nozzleCreateMenuStandard = new ui::Menu(nozzleCreateMenu, "Standard Nozzle Parameter");
                    outputField_->setText("Set the parameters of the standard nozzle");
                    ui::EditField* subMenuSprayAngle_ = new ui::EditField(nozzleCreateMenuStandard, "sprayAngle_");
                    subMenuSprayAngle_->setText("Spray Angle");
                    subMenuSprayAngle_->setCallback([this](const std::string &cmd){
                        manager()->update();

                        try
                        {
                            sprayAngle_ = stof(cmd);

                        }//try

                        catch(const std::invalid_argument& ia)
                        {
                            outputField_->setText("Value of Spray Angle must be an integer/a float");
                            std::cerr << "Invalid argument: " << ia.what() << std::endl;
                            sprayAngle_ = 0;
                        }//catch

                    });
                    ui::EditField* subMenuDecoy_ = new ui::EditField(nozzleCreateMenuStandard, "decoy_");
                    subMenuDecoy_->setText("Decoy");
                    subMenuDecoy_->setCallback([this](const std::string &cmd){
                        manager()->update();
                        decoy_ = cmd.c_str();
                    });
                    ui::EditField* subMenuNozzlename_ = new ui::EditField(nozzleCreateMenuStandard, "nozzlename_");
                    subMenuNozzlename_->setText("Nozzle Name");
                    subMenuNozzlename_->setCallback([this](const std::string &cmd){
                        manager()->update();
                        nozzleNameField_ = cmd;
                    });
                    ui::Action* accept = new ui::Action(nozzleCreateMenuStandard, "acceptStandard");
                    accept->setText("Accept");
                    accept->setCallback([this](){
                        createAndRegisterStandardNozzle();
                    creating = false;
                    //delete tempMenu;
                    nozzleCreateMenuStandard->setVisible(false);
                    nozzleCreateMenu->setVisible(false);
                    outputField_->setText("Help Field");
                });
                }
            });
            }
        }
        else std::cout << "Finish creating of the previous nozzle first" << std::endl;
    });

    remove_ = new ui::Action(sprayMenu_, "removeNozzle");
    remove_->setText("Remove Nozzle");
    remove_->setCallback([this](){
        if(nM->getNozzle(nozzleID) != 0 && editing == false)
        {
            nozzleIDL->select(nozzleID, false);

            auto selList = nozzleIDL->items();
            selList[nozzleID] = "deleted";
            nozzleIDL->setList(selList);;
            editNozzle = nM->getNozzle(nozzleID);
            nM->removeNozzle(nozzleID);
        }
        else
        {
            std::cout << "The nozzle doesn't exist or is already deleted" << std::endl;
            if(editing == true)
                outputField_->setText("Close Editing Tab");
            else
                outputField_->setText("Nozzle doesn't exist");
        }
    });

    //numField = new ui::Label(sprayMenu_, "Save Parameters");

    newGenCreate_ = new ui::EditField(sprayMenu_, "newGenCreate");
    newGenCreate_->setText("Gen Creating Rate");
    newGenCreate_->setValue(parser::instance()->getNewGenCreate());
    newGenCreate_->setCallback([this](const std::string &cmd){
        manager()->update();

        try
        {
            parser::instance()->setNewGenCreate(stoi(cmd));
            printf("%i\n", parser::instance()->getNewGenCreate());

        }//try

        catch(const std::invalid_argument& ia)
        {
            outputField_->setText("Value of Gen Creating Rate must be an integer");
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
        }
    });
    ui::Action* bbEdit = new ui::Action(sprayMenu_,"Edit BB");
    bbEdit->setCallback([this](){
        if(bbEditMenu != nullptr)
            bbEditMenu->setVisible(true);
        else
        {
            bbEditMenu = new ui::Menu(sprayMenu_, "BoundingBox Editor");

            ui::EditField* xBB = new ui::EditField(bbEditMenu, "BoundingBox X");
            xBB->setValue(nM->getBoundingBox().x());
            xBB->setCallback([this](const std::string &cmd){
                manager()->update();

                try
                {
                    osg::Vec3 tempBoundingBox = nM->getBoundingBox();
                    tempBoundingBox.x() = stof(cmd);
                    nM->setBoundingBox(tempBoundingBox);

                }//try

                catch(const std::invalid_argument& ia)
                {
                    outputField_->setText("Value of BoundingBox X must be an integer");
                    std::cerr << "Invalid argument: " << ia.what() << std::endl;
                }
            });

            ui::EditField* yBB = new ui::EditField(bbEditMenu, "BoundingBox Y");
            yBB->setValue(nM->getBoundingBox().x());
            yBB->setCallback([this](const std::string &cmd){
                manager()->update();

                try
                {
                    osg::Vec3 tempBoundingBox = nM->getBoundingBox();
                    tempBoundingBox.y() = stof(cmd);
                    nM->setBoundingBox(tempBoundingBox);

                }//try

                catch(const std::invalid_argument& ia)
                {
                    outputField_->setText("Value of BoundingBox Y must be an integer");
                    std::cerr << "Invalid argument: " << ia.what() << std::endl;
                }
            });

            ui::EditField* zBB = new ui::EditField(bbEditMenu, "BoundingBox Z");
            zBB->setValue(nM->getBoundingBox().x());
            zBB->setCallback([this](const std::string &cmd){
                manager()->update();

                try
                {
                    osg::Vec3 tempBoundingBox = nM->getBoundingBox();
                    tempBoundingBox.z() = stof(cmd);
                    nM->setBoundingBox(tempBoundingBox);

                }//try

                catch(const std::invalid_argument& ia)
                {
                    outputField_->setText("Value of BoundingBox Z must be an integer");
                    std::cerr << "Invalid argument: " << ia.what() << std::endl;
                }
            });

            ui::Action* closeBBEdit = new ui::Action(bbEditMenu, "Close Tab");
            closeBBEdit->setCallback([this](){
                bbEditMenu->setVisible(false);
            });
        }
    });

    scaleFactorParticle = new ui::EditField(sprayMenu_, "Particle Scale");
    scaleFactorParticle->setValue(parser::instance()->getScaleFactor());
    scaleFactorParticle->setCallback([this](const std::string &cmd){
        try
        {
            parser::instance()->setScaleFactor(stof(cmd));

        }//try

        catch(const std::invalid_argument& ia)
        {
            outputField_->setText("Value of Particle Scale must be an integer");
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
        }
    });




    ui::Action* resetScene = new ui::Action(sprayMenu_, "resetScene");
    resetScene->setText("Reset RT Scene");
    resetScene->setCallback([this]()
    {
        raytracer::instance()->removeAllGeometry();                     //resets scene
        nodeVisitorVertex c;                                            //creates new scene
        cover->getObjectsRoot()->accept(c);
    });



    outputField_ = new ui::Label(sprayMenu_,"Help Field");

    scene = new osg::Group;
    cover->getObjectsRoot()->addChild(scene);
    scene->setName("Spray Group");
    testBoxGeode = new osg::Geode;
    testBoxGeode->setName("testBox");

//    float floorHeight = VRSceneGraph::instance()->floorHeight();
//    osg::Box* floorBox = new osg::Box(osg::Vec3(0,0, floorHeight), 3000, 3000, 0.5);
//    osg::TessellationHints *hint = new osg::TessellationHints();
//    hint->setDetailRatio(0.5);
//    osg::ShapeDrawable *floorDrawable = new osg::ShapeDrawable(floorBox, hint);
//    floorDrawable->setColor(osg::Vec4(0, 0.5, 0, 1));
//    floorGeode = new osg::Geode();
//    floorGeode->setName("Floor");
//    floorGeode->addDrawable(floorDrawable);
//    scene->addChild(floorGeode);
//    scene->addChild(testBoxGeode);

    nodeVisitorVertex c;

    cover->getObjectsRoot()->accept(c);
    //scene->accept(c);


    std::cout << std::endl;

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

    //delete editNozzle;
    delete sprayStart_;
    delete save_;
    delete load_;
    delete create_;
    delete remove_;;
    delete pathNameFielddyn_;
    delete fileNameFielddyn_;
    delete nozzleNameFielddyn_;
    //(if creating == true)delete tempMenu;
    delete sprayMenu_;

    return true;
}

bool SprayPlugin::update()
{
    if(sprayStart == true)nM->update();

    return true;
}

void SprayPlugin::createTestBox(osg::Vec3 initPos, osg::Vec3 scale)
{
    osg::Box* testBox = new osg::Box(osg::Vec3(initPos.x(), initPos.z(), initPos.y()), scale.x(), scale.y(), scale.z());
    osg::TessellationHints *hints = new osg::TessellationHints();
    hints->setDetailRatio(0.5);
    osg::ShapeDrawable *boxDrawableTest = new osg::ShapeDrawable(testBox, hints);
    boxDrawableTest->setColor(osg::Vec4(0, 0.5, 0, 1));
    testBoxGeode->addDrawable(boxDrawableTest);
    //cover->getObjectsRoot()->addChild(testBoxGeode);

    idGeo.push_back(raytracer::instance()->createCube(initPos, scale));
}

void SprayPlugin::createAndRegisterImageNozzle()
{
    class nozzle* temporary = nM->createImageNozzle(nozzleNameField_.c_str(),
                                                    pathNameField_,
                                                    fileNameField_);
    if(temporary != NULL)
    {
        temporary->registerLabel();

        nozzleIDL->append(temporary->getName());
    }
}

void SprayPlugin::createAndRegisterStandardNozzle()
{
    class nozzle* temporary = nM->createStandardNozzle(nozzleNameField_.c_str(),
                                                       sprayAngle_,
                                                       decoy_);
    temporary->registerLabel();

    nozzleIDL->append(temporary->getName());

}

void SprayPlugin::createTestBox(osg::Vec3 initPos, osg::Vec3 scale, bool manual)
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
    //cover->getObjectsRoot()->addChild(testBoxGeode);

    //idGeo.push_back(raytracer::instance()->createCube(initPos, scale));
}


COVERPLUGIN(SprayPlugin)
