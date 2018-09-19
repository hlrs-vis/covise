/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "nozzle.h"

nozzle::nozzle(osg::Matrix initialMat, float size, std::string nozzleName/*, osg::Node* geometryNode = nullptr*/):
    coVR3DTransRotInteractor(initialMat,size,coInteraction::ButtonA,"Menu",nozzleName.c_str(),coInteraction::Medium)
{
    nozzleName_ = nozzleName;
    geode_ = new osg::Geode;
    geode_->setName(nozzleName);

    prevEmissionRate = parser::instance()->getEmissionRate();
    minimum = parser::instance()->getMinimum();
    deviation = parser::instance()->getDeviation();


    transform_ = new osg::MatrixTransform;

    float t[] = {1,0,0,0,
                 0,1,0,0,
                 0,0,1,0,
                 0,0,0,1
                };
    osg::Matrix baseTransform;
    baseTransform.set(t);
    transform_->setMatrix(baseTransform);

    nozzleScale = new osg::MatrixTransform();
    baseTransform.makeIdentity();
    baseTransform.makeScale(osg::Vec3(cover->getScale(), cover->getScale(), cover->getScale()));
    nozzleScale->setMatrix(baseTransform);

    particleCount_ = parser::instance()->getReqParticles();

    cone_ = new osg::Cone(osg::Vec3(0,0,0),0.1/**cover->getScale()*/, 0.1/**cover->getScale()*/);                      //diameter = lenght = 0.1m, just for rendering purpose
    cone_->setRotation(osg::Quat(-1,0,0,1));
    shapeDrawable_ = new osg::ShapeDrawable(cone_);
    shapeDrawable_->setColor(osg::Vec4(1,1,0,1));
    printf("Adding basic geometry to nozzle\n");    

    geode_->addDrawable(shapeDrawable_);
    createGeometry();
}

nozzle::~nozzle()
{
    if(!genList.empty()){
        for(auto i = genList.begin();i != genList.end(); i++){
            class gen* current = *i;

            delete current;

        }
        genList.clear();
        printf("list cleared!\n");
    }
    printf("Bye!\n");
}

void nozzle::createGeometry()
{
//    osg::BoundingBox bb;
//    bb = cover->getBBox(geode_);

    interactorGroup = new osg::Group;
    for(int i = 0; i < scaleTransform->getNumChildren(); i++)
    {
        interactorGroup->addChild(scaleTransform->getChild(i));
    }
    scaleTransform->setName("transform");
    scaleTransform->removeChild(0,scaleTransform->getNumChildren());
    scaleTransform->addChild(interactorGroup);
    scaleTransform->addChild(nozzleScale);
    nozzleScale->addChild(geode_);
}

void nozzle::display(bool state)
{
    if(displayed && !state)
    {
        displayed = !displayed;
        interactorGroup->setNodeMask(displayed ? 0xffffffff : 0x0);
    }
    else
        if(!displayed && state)
        {
            displayed = !displayed;
            interactorGroup->setNodeMask(displayed ? 0xffffffff : 0x0);
        }
}

void nozzle::createGen()
{
    //Will never be called
    class gen* newGen = new class gen(initPressure_, this);
    newGen->setColor(getColor());
    newGen->setDeviation(deviation);
    newGen->setMinimum(minimum);
    newGen->setRemoveCount(autoremoveCount);
    newGen->setAlpha(alpha);
    newGen->init();

    genList.push_back(newGen);
}

void nozzle::updateGen()
{
    if(prevEmissionRate != parser::instance()->getEmissionRate())
    {
        prevEmissionRate = parser::instance()->getEmissionRate();
        counter = 0;
    }

    for(auto i = genList.begin();i != genList.end(); i++){
        class gen* current = *i;
        if(current->isOutOfBound() == true)
        {
            if(current->displayedTime > 5.0)
            {
                delete current;
                genList.erase(i);
                i = genList.begin();
            }
            else
                current->displayedTime += cover->frameDuration();
        }
        else
        {
            std::clock_t begin = clock();
            //current->updatePos(boundingBox_);
            current->updateAll(boundingBox_);

            std::clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

            printf("elapsed time for updating %f\n", elapsed_secs);
        }
    }
    if(counter == parser::instance()->getEmissionRate())
    {
        createGen();
        counter = 0;
    }
    counter++;
}

void nozzle::updateColor()
{
    for(auto i = genList.begin();i != genList.end(); i++)
    {
        class gen* current = *i;
        current->setColor(currentColor_);
    }
}

void nozzle::save(std::string pathName, std::string fileName)
{
    FILE* saving = new FILE;
    char ptr[1000];

    saving = fopen((pathName+fileName).c_str(), "a");
    fputs("\n", saving);

    fputs("name = ", saving);
    fputs(nozzleName_.c_str(), saving);
    fputs("\n", saving);

    fputs("position = \n", saving);
    osg::Matrix nozzleMatrix = this->getMatrix();

    sprintf(ptr, "%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",
            nozzleMatrix.ptr()[0],nozzleMatrix.ptr()[1],nozzleMatrix.ptr()[2],nozzleMatrix.ptr()[3],
            nozzleMatrix.ptr()[4],nozzleMatrix.ptr()[5],nozzleMatrix.ptr()[6],nozzleMatrix.ptr()[7],
            nozzleMatrix.ptr()[8],nozzleMatrix.ptr()[9],nozzleMatrix.ptr()[10],nozzleMatrix.ptr()[11],
            nozzleMatrix.ptr()[12],nozzleMatrix.ptr()[13],nozzleMatrix.ptr()[14],nozzleMatrix.ptr()[15]);

    fputs(ptr,saving);
    fputs("\n",saving);

    fputs("type = NULL\n", saving);
    fputs("\n",saving);

    fputs("minimum = ", saving);
    sprintf(ptr, "%f\n", minimum);

    fputs("deviation = ", saving);
    sprintf(ptr, "%f\n", deviation);
    fputs(ptr,saving);
    fputs("\n",saving);

    fputs("-", saving);
    fclose(saving);

    printf("Nozzle configuration saved!\n");
}

void nozzle::setID(int ID)
{
    if(initialized == false)
    {
        nozzleID = ID;
        initialized = true;
    }
    else
        std::cout << "ID was already set" << std::endl;
}

int nozzle::getID()
{
    return nozzleID;
}

void nozzle::keepSize()
{
    osg::Matrix m;
    m.makeScale(1, 1, 1);
    scaleTransform->setMatrix(m);
}

void nozzle::autoremove(bool state)
{
    for(auto i = genList.begin();i != genList.end(); i++)
    {
        class gen* current = *i;
        if(state)
        {
            autoremoveCount = 1.5;
            current->setRemoveCount(autoremoveCount);
        }
        else
        {
            autoremoveCount = 1.0;
            current->setRemoveCount(autoremoveCount);
        }
    }
}

void nozzle::setNozzleGeometryNode(osg::Node* node)
{
    if(node != nullptr)
    {
    osg::Matrix coverToNode;
    coverToNode.makeIdentity();
    auto parentList = node->getParentalNodePaths();

    for(int i = 0; i < /*parentList.size()*/1; i++)
    {
        auto pl = parentList[i];
        int itr = 0;
        while(pl[itr]->getName().compare("OBJECTS_ROOT") != 0)
            itr++;
        for( ; itr < pl.size(); itr++)
        {
            if (auto nozzleMatrixTransform = dynamic_cast<osg::MatrixTransform *>(pl[itr]))
            {
                coverToNode *= nozzleMatrixTransform->getMatrix();
            }
        }
    }


    if (auto geode = dynamic_cast<osg::Geode *>(node))
    {
        osg::Group* p = node->getParent(0);
        nozzleScale->addChild(geode);
        p->removeChild(node);
    }

    if (auto matrixTransform = dynamic_cast<osg::MatrixTransform*>(node))
    {
        //nozzleScale->removeChildren(0,nozzleScale->getNumChildren());
        for(int i = 0; i < matrixTransform->getNumChildren(); i++)
        {
            if (auto geode = dynamic_cast<osg::Geode*>(matrixTransform->getChild(i)))
                nozzleScale->addChild(geode);
            //nozzleScale->addChild(matrixTransform->getChild(i));
        }
        matrixTransform->removeChildren(0, matrixTransform->getNumChildren());
        coverToNode = scaleTransform->getMatrix()*coverToNode;
        scaleTransform->setMatrix(coverToNode);
        osg::Group* p = node->getParent(0);
        p->removeChild(node);
    }

    //osg::BoundingBox bb = cover->getBBox(nozzleScale);
    scaleTransform->dirtyBound();
    scaleTransform->getBound();
    nozzleScale->dirtyBound();
    nozzleScale->getBound();
    }
}

void nozzle::setScale(float newScale)
{
    scale = newScale;
    osg::Matrix scaleMatrix = nozzleScale->getMatrix();
    scaleMatrix.makeScale(newScale, newScale, newScale);
    nozzleScale->setMatrix(scaleMatrix);
}






standardNozzle::standardNozzle(float sprayAngle, std::string decoy, osg::Matrix initialMat, float size, std::string nozzleName) :
    nozzle(initialMat, size, nozzleName)
{
    if(sprayAngle == 0.00)
    {
        sprayAngle_ = 30;
    }
    else sprayAngle_ = sprayAngle;
    if(decoy.empty())
        decoy_ = "NONE";
    else decoy_ = decoy.c_str();

    stringstream ss;
    ss << sprayAngle_;

    setParam1(ss.str());
    setParam2(decoy_);
    setType("standard");
}

void standardNozzle::createGen()
{
    class standardGen* newGen = new class standardGen(sprayAngle_, decoy_,getInitPressure(), this);
    newGen->init();
    newGen->setDeviation(getDeviation());
    newGen->setMinimum(getMinimum());
    newGen->setRemoveCount(autoremoveCount);
    newGen->setAlpha(getAlpha());
    newGen->seed();
    if(parser::instance()->getSphereRenderType() == 0)
        newGen->setColor(getColor());

    genList.push_back(newGen);
}

void standardNozzle::save(std::string pathName, std::string fileName)
{
    FILE* saving = new FILE;
    char ptr[20];
    char matrixPtr[1000];

    saving = fopen((pathName+fileName).c_str(), "a");
    fputs("\n", saving);

    fputs("name = ", saving);
    fputs(nozzleName_.c_str(), saving);
    fputs("\n", saving);
    fputs("\n", saving);

    fputs("position = \n", saving);
    osg::Matrix nozzleMatrix = this->getMatrix();

    sprintf(matrixPtr, "%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",  nozzleMatrix.ptr()[0],nozzleMatrix.ptr()[1],nozzleMatrix.ptr()[2],nozzleMatrix.ptr()[3],
            nozzleMatrix.ptr()[4],nozzleMatrix.ptr()[5],nozzleMatrix.ptr()[6],nozzleMatrix.ptr()[7],
            nozzleMatrix.ptr()[8],nozzleMatrix.ptr()[9],nozzleMatrix.ptr()[10],nozzleMatrix.ptr()[11],
            nozzleMatrix.ptr()[12],nozzleMatrix.ptr()[13],nozzleMatrix.ptr()[14],nozzleMatrix.ptr()[15]);

    fputs(matrixPtr,saving);
    fputs("\n",saving);

    fputs("type = standard\n", saving);
    char* returnVal = gcvt(sprayAngle_, 5,ptr);
    //fputs("\n", saving);
    fputs("angle = ", saving);
    fputs(ptr, saving);
    fputs("\n", saving);
    fputs("decoy = ", saving);
    fputs(decoy_.c_str(),saving);
    fputs("\n",saving);

    fputs("minimum = ", saving);
    sprintf(ptr, "%f\n", getMinimum());
    fputs(ptr,saving);

    fputs("deviation = ", saving);
    sprintf(ptr, "%f\n", getDeviation());
    fputs(ptr,saving);

    fputs("\n",saving);
    fputs("-", saving);
    fclose(saving);

    printf("Nozzle configuration saved!\n");
}




imageNozzle::imageNozzle(std::string pathName, std::string fileName, osg::Matrix initialMat, float size, std::string nozzleName):
    nozzle(initialMat, size, nozzleName)
{
    if(fileName.empty())
        fileName_ = "Nozzle_1000_4_8_2.bmp";
    else
        fileName_ = fileName;
    pathName_ = pathName;

    samplingPoints = parser::instance()->getReqSamplings();
    particleCount_ = samplingPoints;

    colorThreshold_ = parser::instance()->getColorThreshold();

    if(parser::instance()->getSamplingType().compare("circle") == 0)
    {
        circle = true;
        square = false;
    }

    if(parser::instance()->getSamplingType().compare("square") == 0)
    {
        circle = false;
        square = true;
    }

    if(readImage() == false)
    {
        failed = true;
    }

    setParam1(pathName_);
    setParam2(fileName_);
    setType("image");
}

void imageNozzle::createGen()
{
    class imageGen* newGen = new class imageGen(&iBuf,getInitPressure(), this);
    newGen->init();
    newGen->setDeviation(getDeviation());
    newGen->setMinimum(getMinimum());
    newGen->setRemoveCount(autoremoveCount);
    newGen->setAlpha(getAlpha());
    newGen->seed();
    if(parser::instance()->getSphereRenderType() == 0)
        newGen->setColor(getColor());

    genList.push_back(newGen);
}

void imageNozzle::save(std::string pathName, std::string fileName)
{
    FILE* saving = new FILE;
    char matrixPtr[1000];
    char ptr[20];

    saving = fopen((pathName+fileName).c_str(), "a");
    fputs("\n", saving);

    fputs("name = ", saving);
    fputs(nozzleName_.c_str(), saving);
    fputs("\n", saving);

    fputs("position = \n", saving);
    osg::Matrix nozzleMatrix = this->getMatrix();

    sprintf(matrixPtr, "%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",  nozzleMatrix.ptr()[0],nozzleMatrix.ptr()[1],nozzleMatrix.ptr()[2],nozzleMatrix.ptr()[3],
            nozzleMatrix.ptr()[4],nozzleMatrix.ptr()[5],nozzleMatrix.ptr()[6],nozzleMatrix.ptr()[7],
            nozzleMatrix.ptr()[8],nozzleMatrix.ptr()[9],nozzleMatrix.ptr()[10],nozzleMatrix.ptr()[11],
            nozzleMatrix.ptr()[12],nozzleMatrix.ptr()[13],nozzleMatrix.ptr()[14],nozzleMatrix.ptr()[15]);

    fputs(matrixPtr,saving);
    fputs("\n",saving);

    fputs("type = image\n", saving);
    fputs("pathname = ", saving);
    fputs(pathName_.c_str(), saving);
    fputs("\n", saving);
    fputs("filename = ", saving);
    fputs(fileName_.c_str(), saving);
    fputs("\n", saving);

    fputs("minimum = ", saving);
    sprintf(ptr, "%f\n", getMinimum());
    fputs(ptr,saving);

    fputs("deviation = ", saving);
    sprintf(ptr, "%f\n", getDeviation());
    fputs(ptr,saving);
    fputs("\n",saving);

    fputs("\n",saving);
    fputs("-", saving);

    fclose(saving);

    printf("Nozzle configuration saved!\n");
}

bool imageNozzle::readImage()
{
    osg::Image* image = osgDB::readImageFile(fileName_);

    if(image == NULL)
    {
        std::cout << "Image wasn't readable" << std::endl;
        return false;
    }

    int corner_points[] = {0,0,0,0};
    int* points_(new int[samplingPoints]);
    /**********************************************************************************/
    //Testweise erstellt zum Einlesen von Namen

    std::string line;

    std::stringstream nameStream(fileName_);

    std::getline(nameStream, line, '_');                        //Name of the nozzle
    std::cout << line << std::endl;

    std::getline(nameStream, line, '_');
    float height = stof(line);

    std::getline(nameStream, line, '_');
    pixel_to_mm_ = stof(line)/10;

    std::getline(nameStream, line, '_');
    pixel_to_flow_ = stof(line);

    std::cout << height << " " << pixel_to_mm_ << " " << pixel_to_flow_ << std::endl;

    /**********************************************************************************/


    if(image->isDataContiguous())
        printf("Data is contiguous\n");
    else
        printf("Data is not contiguous!\n");

    int colorDepth = image->getTotalDataSize()/(image->s()*image->t());
    colorDepth_ = colorDepth;
    int s_int = image->s()*colorDepth;
    int t_int = image->t()*colorDepth;

    int center = (image->s()*0.5*image->t()+image->s()*0.5)*colorDepth;
    int center_width = (center%s_int)/colorDepth;
    int center_height = (center-center_width)/t_int;

    int point_width = 0;
    int point_height = 0;

    /***********************************************************************************************/
    //Defines corner points of the read image

    if(square)
    {
        std::cout << "square" << std::endl;
        for(int i = 0; i<4;i++){
            int random_point = 0;
            if(i == 0){
                random_point = center-0.5*s_int;
                int count = 3;
                while(image->data()[random_point]<colorThreshold_){
                    random_point = center-image->s()*0.5+count;
                    count+=3;
                    if(random_point > center)
                    {
                        random_point = center-0.5*s_int;
                        printf("Nothing found in negative x-direction!\n");
                        break;
                    }
                }
            }
            if(i == 1){
                random_point = center+0.5*s_int;
                int count = 3;
                while(image->data()[random_point]<colorThreshold_)
                {
                    random_point = center+image->s()*0.5-count;
                    count+=3;
                    if(random_point < center)
                    {
                        random_point = center+0.5*s_int;
                        printf("Nothing found in positive x-direction!\n");
                        break;
                    }
                }
            }
            if(i == 2){
                random_point = 1.5*s_int;
                while(image->data()[random_point]<colorThreshold_){
                    random_point += image->s();
                    if(random_point > center)
                    {
                        random_point = 1.5*s_int;
                        printf("Nothing found in positive y-direction!\n");
                        break;
                    }
                }
            }
            if(i == 3){
                random_point = image->getTotalDataSize()-image->s()*1.5;
                while(image->data()[random_point]<colorThreshold_){
                    random_point -= image->s();
                    if(random_point < center)
                    {
                        random_point = image->getTotalDataSize()-image->s()*1.5;
                        printf("Nothing found in negative y-direction!\n");
                        break;
                    }

                }
            }
            corner_points[i] = random_point;

        }

        int delta_1 = abs(center-corner_points[0]);
        int delta_2 = abs(center-corner_points[1]);
        int delta_3 = abs(center-corner_points[2]);
        int delta_4 = abs(center-corner_points[3]);

        if(delta_3>delta_4){
            corner_points[3] = center+delta_3;
        }
        else corner_points[2] = center-delta_4;

        if(delta_1>delta_2){
            corner_points[0] = corner_points[2]-delta_1;
            corner_points[1] = corner_points[2]+delta_1;
            corner_points[2] = corner_points[3]-delta_1;
            corner_points[3] += delta_1;
        }
        else{
            corner_points[0] = corner_points[2]-delta_2;
            corner_points[1] = corner_points[2]+delta_2;
            corner_points[2] = corner_points[3]-delta_2;
            corner_points[3] += delta_2;
        }

        delta_1 = corner_points[1]-corner_points[0];                                //width
        delta_2 = (corner_points[2]-corner_points[0])/s_int;                        //height

    /***********************************************************************************************/
    //Based on the amount of samplings, specific points are chosen from the relevant area

        if(samplingPoints < 10*delta_2)
            samplingPoints = 10*delta_2;                          //This is needed to get at least minimum accuracy
        else
            if(samplingPoints > image->getTotalDataSize())
                samplingPoints = image->getTotalDataSize();
        points_ = new int[samplingPoints];

        int ppl = samplingPoints/(delta_2);
        int steppingPerLine = delta_1/(ppl-1);
        if(steppingPerLine%3 != 0)
            steppingPerLine -= steppingPerLine%3+3;

        std::cout << ppl << " " <<steppingPerLine << std::endl;
        std::cout << delta_1 << " " << delta_2 << std::endl;

//        for(int i = 0; i < delta_2; i++)
//        {
//            for(int j = 0; j < ppl;j++)
//            {
//                if(image->data()[corner_points[0]+steppingPerLine*j+s_int*i] == 0)
//                    continue;
//                else
//                {
//                    points_[count] = corner_points[0]+steppingPerLine*j+s_int*i;
//                    count++;
//                }
//            }
//        }

        int count = 0;
        for(int i = 0; i < delta_2; i++)
        {
            for(int j = 0; j< ppl; j++)
            {
                if(image->data()[posToCount((center_width-delta_1/2+j*steppingPerLine), (center_height-delta_2/2+i), s_int)] == 0)
                    continue;
                else
                {
                    points_[count] = posToCount((center_width-delta_1/2+j*steppingPerLine), (center_height-delta_2/2+i), s_int);
                    count++;
                }
            }
        }

        samplingPoints = count;
        std::cout << count << std::endl;
    }

    if(circle)
    {
        std::cout << "circle" << std::endl;

        int nrOfCircles = 15;
        int samplingRadius = getMin(s_int*0.5-3,t_int*0.5-3);

        float div = 0;
        for(int i = 0;i<nrOfCircles;i++)
        {
            div += ((float)nrOfCircles-i)/(float)nrOfCircles;
        }

        float nBegin = samplingPoints/div;

        int count = 0;
        for(int j = 0; j<nrOfCircles;j++)
        {
            float curPoints = nBegin*(1-(float)j/(float)nrOfCircles);
            float curRadius = samplingRadius*(1-(float)j/(float)nrOfCircles)/colorDepth;
            float step = 360/curPoints;

            for(int i = 0; i<(int)curPoints;i++)
            {
                int x = curRadius*cos(step*i*Pi/180);
                int y = curRadius*sin(step*i*Pi/180);

                if(image->data()[posToCount((center_width+x),(center_height+y),s_int)] == 0)
                    continue;
                else
                {
                    points_[count] = posToCount((center_width+x),(center_height+y),s_int);
                    count++;
                }

            }
        }

        printf("%i\n", count);
        samplingPoints = count;
    }

    int index = 0;


    /***********************************************************************************************/
    //Angles for later purpose are defined here

    int deltaWidth = 0;
    int deltaHeight = 0;
    float angleToNozzle = 0;
    float angleToPosition = 0;
    float particleFlow = 0;


    iBuf.centerX = center_width;
    iBuf.centerY = center_height;
    iBuf.samplingPoints = samplingPoints;
    iBuf.dataBuffer = new float[samplingPoints*5];

    for(int count = 0; count < samplingPoints; count++){

        index = points_[count];       

        point_width = (index%s_int)/colorDepth;
        point_height = (index-point_width)/t_int;

        deltaWidth = point_width-center_width;         //Particle width - center +2
        deltaHeight = point_height-center_height;       //Particle height - center +3

        float hypotenuse = sqrt(pow(deltaWidth*pixel_to_mm_,2)+pow(deltaHeight,2));
        angleToNozzle = atan2(hypotenuse,height);
        //angleToNozzle = atan2(height, hypotenuse);

        angleToPosition = atan2(deltaWidth,deltaHeight);
        //Angle between height and width of particle - center

        particleFlow = 0.20*(  image->data()[index]+
                               image->data()[index+colorDepth_]+
                               image->data()[index-colorDepth_]+
                               image->data()[index+s_int]+
                               image->data()[index-s_int]
                                )*pixel_to_flow_/255;
        particleFlow = 1;
        //Flow

        iBuf.dataBuffer[count*5] = angleToNozzle;
        iBuf.dataBuffer[count*5+1] = angleToPosition;
        iBuf.dataBuffer[count*5+2] = deltaWidth;
        iBuf.dataBuffer[count*5+3] = deltaHeight;
        iBuf.dataBuffer[count*5+4] = particleFlow;

    };

    std::cout << "Reading image completed!\n" << std::endl;

    return true;
}
