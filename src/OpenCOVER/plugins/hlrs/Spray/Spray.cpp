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
    nM->init();
    idGeo.clear();
    raytracer::instance()->init();


    //Creation of the interface

    sprayMenu_ = new ui::Menu("Spray", this);
    sprayMenu_->setText("Spray");

    currentNozzle_ = new ui::EditField(sprayMenu_,"currentNozzle");
    currentNozzle_->setText("Current Nozzle ID");
    currentNozzle_->setValue(nozzleID);
    currentNozzle_->setCallback([this](const std::string &cmd){
        manager()->update();

        try
        {
        nozzleID = stof(cmd);

        }//try

        catch(const std::invalid_argument& ia)
        {
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
            nozzleID = -1;
        }//catch

        if(editing == false)editNozzle = nM->getNozzle(nozzleID);
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

                acceptEdit_ = new ui::Action(nozzleEditMenu_, "acceptEdit");
                acceptEdit_->setText("Accept");
                acceptEdit_->setCallback([this](){
                    manager()->update();
                    editNozzle->setColor(newColor);
                    editNozzle->setInitPressure(pressureSlider_->value());
                    nozzleEditMenu_->remove(red_);
                    nozzleEditMenu_->remove(blue_);
                    nozzleEditMenu_->remove(green_);
                    nozzleEditMenu_->remove(alpha_);
                    nozzleEditMenu_->remove(pressureSlider_);
                    editing = false;
                    std::cout << "Editing done" << std::endl;
                    delete nozzleEditMenu_;

                });

//#if TESTING
                testMenu = new ui::Menu(nozzleEditMenu_, "tester");
                testMenu->setText("Controller");

                memMat.makeIdentity();

//                rotXneg = new ui::Action(testMenu, "rotXn");
//                rotXneg->setText("Rotate X negative");
//                rotXneg->setCallback([this](){
//                    osg::Vec3 trans = editNozzle->getMatrix().getTrans();
//                    osg::Quat a(-1,0,0,1);
//                    memMat.makeIdentity();
//                    memMat.setRotate(a);
//                    memMat.setTrans(trans);
//                    editNozzle->updateTransform(memMat);

//                });

//                rotXpos = new ui::Action(testMenu, "rotXp");
//                rotXpos->setText("Rotate X positive");
//                rotXpos->setCallback([this](){
//                    osg::Vec3 trans = editNozzle->getMatrix().getTrans();
//                    osg::Quat a(1,0,0,1);
//                    memMat.makeIdentity();
//                    memMat.setRotate(a);
//                    memMat.setTrans(trans);
//                    editNozzle->updateTransform(memMat);

//                });

//                rotYneg = new ui::Action(testMenu, "rotYn");
//                rotYneg->setText("Rotate Y negative");
//                rotYneg->setCallback([this](){
//                    osg::Vec3 trans = editNozzle->getMatrix().getTrans();
//                    osg::Quat a(0,-1,0,1);
//                    memMat.makeIdentity();
//                    memMat.setRotate(a);
//                    memMat.setTrans(trans);
//                    editNozzle->updateTransform(memMat);

//                });

//                rotYpos = new ui::Action(testMenu, "rotYp");
//                rotYpos->setText("Rotate Y positive");
//                rotYpos->setCallback([this](){
//                    osg::Vec3 trans = editNozzle->getMatrix().getTrans();
//                    osg::Quat a(0,1,0,1);
//                    memMat.makeIdentity();
//                    memMat.setRotate(a);
//                    memMat.setTrans(trans);
//                    editNozzle->updateTransform(memMat);

//                });

//                rotZneg = new ui::Action(testMenu, "rotZn");
//                rotZneg->setText("Rotate Z negative");
//                rotZneg->setCallback([this](){
//                    osg::Vec3 trans = editNozzle->getMatrix().getTrans();
//                    osg::Quat a(0,0,-1,1);
//                    memMat.makeIdentity();
//                    memMat.setRotate(a);
//                    memMat.setTrans(trans);
//                    editNozzle->updateTransform(memMat);

//                });

//                rotZpos = new ui::Action(testMenu, "rotZp");
//                rotZpos->setText("Rotate Z positive");
//                rotZpos->setCallback([this](){
//                    osg::Vec3 trans = editNozzle->getMatrix().getTrans();
//                    osg::Quat a(0,0,1,1);
//                    memMat.makeIdentity();
//                    memMat.setRotate(a);
//                    memMat.setTrans(trans);
//                    editNozzle->updateTransform(memMat);

//                });

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
                    manager()->update();

                    try
                    {
                        transMat.x() = stof(cmd);
                        memMat = editNozzle->getMatrix();
                        memMat.setTrans(transMat);
                        //memMat = editNozzle->getMatrixTransform()->getMatrix()*memMat;
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
                moveY->setCallback([this](const std::string &cmd){
                    manager()->update();

                    try
                    {
                        transMat.y() = stof(cmd);
                        memMat = editNozzle->getMatrix();
                        memMat.setTrans(transMat);
                        //memMat = editNozzle->getMatrixTransform()->getMatrix()*memMat;
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
                moveZ->setCallback([this](const std::string &cmd){
                    manager()->update();

                    try
                    {
                        transMat.z() = stof(cmd);
                        memMat = editNozzle->getMatrix();
                        memMat.setTrans(transMat);
                        //memMat = editNozzle->getMatrixTransform()->getMatrix()*memMat;
                        editNozzle->updateTransform(memMat);

                    }//try

                    catch(const std::invalid_argument& ia)
                    {
                        std::cerr << "Invalid argument: " << ia.what() << std::endl;
                        transMat.z() = 0;
                    }
                });

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
        nM->saveNozzle(pathNameField_, fileNameField_);
    });

    load_ = new ui::Action(sprayMenu_, "Load");
    load_->setText("Load");
    load_->setCallback([this](){
        manager()->update();
        nM->loadNozzle(pathNameField_.c_str(), fileNameField_.c_str());
        while(nM->checkAll() != NULL)
        {
            nozzle* temporary = nM->checkAll();
            ui::Label* tempLabel = temporary->registerLabel();

            std::stringstream ss;
            ss << temporary->getName() << " " << temporary->getID();
            tempLabel =  new ui::Label(sprayMenu_, ss.str());
        }
    });

    create_ = new ui::Action(sprayMenu_, "createNozzle");
    create_->setText("Create Nozzle");
    create_->setCallback([this](){

        if(creating == false)
        {
            creating = true;

            tempMenu = new ui::Menu(sprayMenu_, "Type of nozzle");

            ui::Action* createImage = new ui::Action(tempMenu, "image");
            createImage->setText("Create image nozzle");
            createImage->setCallback([this](){
                ui::Menu* tempMenu2 = new ui::Menu(tempMenu, "Image Nozzle Parameters");
                ui::EditField* subMenuPathname_ = new ui::EditField(tempMenu2, "pathname_");
                subMenuPathname_->setText("Path Name");
                subMenuPathname_->setCallback([this](const std::string &cmd){
                    manager()->update();
                    pathNameField_ = cmd;
                });
                ui::EditField* subMenuFilename_ = new ui::EditField(tempMenu2, "filename_");
                subMenuFilename_->setText("File Name");
                subMenuFilename_->setCallback([this](const std::string &cmd){
                    manager()->update();
                    fileNameField_ = cmd;
                });
                ui::EditField* subMenuNozzlename_ = new ui::EditField(tempMenu2, "nozzlename_");
                subMenuNozzlename_->setText("Nozzle Name");
                subMenuNozzlename_->setCallback([this](const std::string &cmd){
                    manager()->update();
                    nozzleNameField_ = cmd;
                });

                ui::Action* accept = new ui::Action(tempMenu2, "accept");
                accept->setText("Accept");
                accept->setCallback([this](){
                    createAndRegisterImageNozzle();
                    creating = false;
                    delete tempMenu;
                });


            });

            ui::Action* createStandard = new ui::Action(tempMenu, "standard");
            createStandard->setText("Create standard nozzle");
            createStandard->setCallback([this](){
                ui::Menu* tempMenu2 = new ui::Menu(tempMenu, "Standard Nozzle Parameter");
                ui::EditField* subMenuSprayAngle_ = new ui::EditField(tempMenu2, "sprayAngle_");
                subMenuSprayAngle_->setText("Spray Angle");
                subMenuSprayAngle_->setCallback([this](const std::string &cmd){
                    manager()->update();

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
                ui::EditField* subMenuDecoy_ = new ui::EditField(tempMenu2, "decoy_");
                subMenuDecoy_->setText("Decoy");
                subMenuDecoy_->setCallback([this](const std::string &cmd){
                    manager()->update();
                    decoy_ = cmd.c_str();
                });
                ui::EditField* subMenuNozzlename_ = new ui::EditField(tempMenu2, "nozzlename_");
                subMenuNozzlename_->setText("Nozzle Name");
                subMenuNozzlename_->setCallback([this](const std::string &cmd){
                    manager()->update();
                    nozzleNameField_ = cmd;
                });
                ui::Action* accept = new ui::Action(tempMenu2, "accept");
                accept->setText("Accept");
                accept->setCallback([this](){
                    createAndRegisterStandardNozzle();
                    creating = false;
                    delete tempMenu;
                });
            });

        }
        else std::cout << "Finish creating of the previous nozzle first" << std::endl;
    });

    remove_ = new ui::Action(sprayMenu_, "removeNozzle");
    remove_->setText("Remove Nozzle");
    remove_->setCallback([this](){
        if(nM->getNozzle(nozzleID) != 0 && editing == false)
        {
            editNozzle = nM->getNozzle(nozzleID);
            ui::Label* tempLabel = editNozzle->getLabel();
            printf("here");
            sprayMenu_->remove(tempLabel);
            //delete tempLabel;
            printf("here");
            nM->removeNozzle(nozzleID);
        }
        else std::cout << "The nozzle doesn't exist or is already deleted" << std::endl;
    });

    numField = new ui::Label(sprayMenu_, "Save Parameters");

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

    newGenCreate_ = new ui::EditField(sprayMenu_, "newGenCreate");
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
            std::cerr << "Invalid argument: " << ia.what() << std::endl;
        }
    });


    scene = new osg::Group;
    cover->getObjectsRoot()->addChild(scene);
    scene->setName("Spray Group");
    testBoxGeode = new osg::Geode;
    testBoxGeode->setName("testBox");

    float floorHeight = VRSceneGraph::instance()->floorHeight();
    osg::Box* floorBox = new osg::Box(osg::Vec3(0,0, floorHeight), 3000, 3000, 1);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    osg::ShapeDrawable *floorDrawable = new osg::ShapeDrawable(floorBox, hint);
    floorDrawable->setColor(osg::Vec4(0, 0.5, 0, 1));
    floorGeode = new osg::Geode();
    floorGeode->setName("Floor");
    floorGeode->addDrawable(floorDrawable);
    scene->addChild(floorGeode);
    scene->addChild(testBoxGeode);

    createTestBox(osg::Vec3(0,0,1000), osg::Vec3(300,300,300), false);
    createTestBox(osg::Vec3(500,100,1000), osg::Vec3(300,300,300), false);



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

    delete currentNozzle_;
    delete sprayStart_;
    delete save_;
    delete load_;
    delete create_;
    delete remove_;
    delete numField;
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
        ui::Label* tempLabel = temporary->registerLabel();
        std::stringstream ss;
        ss << temporary->getName() << " " << temporary->getID();
        tempLabel = new ui::Label(sprayMenu_, ss.str());
    }
}

void SprayPlugin::createAndRegisterStandardNozzle()
{
    class nozzle* temporary = nM->createStandardNozzle(nozzleNameField_.c_str(),
                                                       sprayAngle_,
                                                       decoy_);
    ui::Label* tempLabel = temporary->registerLabel();

    std::stringstream ss;
    ss << temporary->getName() << " " << temporary->getID();
    tempLabel =  new ui::Label(sprayMenu_, ss.str());
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
    normals->push_back(osg::Vec3(0,-1,0));
    geom->setNormalArray(normals);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,8));


    testBoxGeode->addDrawable(geom);
    //cover->getObjectsRoot()->addChild(testBoxGeode);

    //idGeo.push_back(raytracer::instance()->createCube(initPos, scale));
}


COVERPLUGIN(SprayPlugin)
