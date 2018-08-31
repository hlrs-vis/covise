/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "nozzle.h"

nozzle::nozzle(osg::Matrix initialMat, float size, std::string nozzleName):
    coVR3DTransRotInteractor(initialMat,size,coInteraction::ButtonA,"Menu",nozzleName.c_str(),coInteraction::Medium)
{
    nozzleName_ = nozzleName;
    geode_ = new osg::Geode;
    geode_->setName(nozzleName);

    prevGenCreate = parser::instance()->getNewGenCreate();
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
    //scaleTransform->setMatrix(baseTransform);

    particleCount_ = parser::instance()->getReqParticles();

    box_ = new osg::Box(osg::Vec3(0,0,0),0.1);                      //diameter = lenght = 10, just for rendering purpose
    shapeDrawable_ = new osg::ShapeDrawable(box_);
    shapeDrawable_->setColor(osg::Vec4(1,1,0,1));
    printf("Adding basic geometry to nozzle\n");

    geode_->addDrawable(shapeDrawable_);

    createGeometry();
}

nozzle::~nozzle(){
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

void nozzle::createGeometry(){
    osg::BoundingBox bb;
    bb = cover->getBBox(geode_);
    scaleTransform->setName("transform");
    scaleTransform->removeChild(0,scaleTransform->getNumChildren());
    scaleTransform->addChild(geode_);
}

void nozzle::createGen(){
    class gen* newGen = new class gen(initPressure_, this);
    newGen->setColor(getColor());
    newGen->setDeviation(deviation);
    newGen->setMinimum(minimum);
    newGen->setRemoveCount(autoremoveCount);
    newGen->init();

    genList.push_back(newGen);
}

void nozzle::updateGen(){

    if(prevGenCreate != parser::instance()->getNewGenCreate())
    {
        prevGenCreate = parser::instance()->getNewGenCreate();
        counter = 0;
    }

    for(auto i = genList.begin();i != genList.end(); i++){
        class gen* current = *i;
        if(current->isOutOfBound() == true){
            printf("generation out of bound\n");
            delete current;
            genList.erase(i);
            i = genList.begin();
        }
        else{
            current->updatePos(boundingBox_);
        }
    }
    if(counter == parser::instance()->getNewGenCreate()){
        createGen();
        counter = 0;
        printf("New gen created\n");
    }
    counter++;
}

void nozzle::updateColor(){
    for(auto i = genList.begin();i != genList.end(); i++){
        class gen* current = *i;
        current->setColor(currentColor_);
    }
}

void nozzle::save(std::string pathName, std::string fileName){
    FILE* saving = new FILE;
    char ptr[1000];

    saving = fopen((pathName+fileName).c_str(), "a");
    fputs("\n", saving);

    fputs("name = ", saving);
    fputs(nozzleName_.c_str(), saving);
    fputs("\n", saving);

    fputs("position = \n", saving);
    osg::Matrix nozzleMatrix = this->getMatrix();

    sprintf(ptr, "%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",  nozzleMatrix.ptr()[0],nozzleMatrix.ptr()[1],nozzleMatrix.ptr()[2],nozzleMatrix.ptr()[3],
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
    else std::cout << "ID was already set" << std::endl;
}

int nozzle::getID()
{
    return nozzleID;
}

void nozzle::keepSize()
{
    osg::Matrix m;
    float size = cover->getScale();
    m.makeScale(size, size, size);
    scaleTransform->setMatrix(m);
}

void nozzle::autoremove(bool state)
{
    for(auto i = genList.begin();i != genList.end(); i++){
        class gen* current = *i;
        if(state)
        {
            autoremoveCount = 1.5;
            current->setRemoveCount(autoremoveCount);
        }
        else
        {
            autoremoveCount = 0.9;
            current->setRemoveCount(0.9);
        }
    }
}





standardNozzle::standardNozzle(float sprayAngle, std::string decoy, osg::Matrix initialMat, float size, std::string nozzleName) :
    nozzle(initialMat, size, nozzleName)
{
    if(sprayAngle == 0)
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

void standardNozzle::createGen(){
    class standardGen* newGen = new class standardGen(sprayAngle_, decoy_,getInitPressure(), this);
    newGen->init();
    //newGen->setColor(getColor());
    newGen->setDeviation(getDeviation());
    newGen->setMinimum(getMinimum());
    newGen->setRemoveCount(autoremoveCount);
    newGen->seed();

    genList.push_back(newGen);
}

void standardNozzle::save(std::string pathName, std::string fileName){
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
        fileName_ = "Nozzle_1000_4_1_2.bmp";
    else
        fileName_ = fileName;
    pathName_ = pathName;

    samplingPoints = particleCount_;
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
        delete this;
        failed = true;
    }

    setParam1(pathName_);
    setParam2(fileName_);
    setType("image");
}

imageNozzle::~imageNozzle()
{

}

void imageNozzle::createGen(){
    class imageGen* newGen = new class imageGen(&iBuf,getInitPressure(), this);
    newGen->init();
    //newGen->setColor(getColor());
    newGen->setDeviation(getDeviation());
    newGen->setMinimum(getMinimum());
    newGen->setRemoveCount(autoremoveCount);
    newGen->seed();

    genList.push_back(newGen);
}

void imageNozzle::save(std::string pathName, std::string fileName){
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

    int* points_(new int[samplingPoints]);

    int corner_points[] = {0,0,0,0};
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

    std::getline(nameStream, line, '.');
    pixel_to_radius_ = stof(line)*0.001;

    std::cout << height << " " << pixel_to_mm_ << " " << pixel_to_flow_ << " " << pixel_to_radius_ << std::endl;

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
    int count = 0;

    /***********************************************************************************************/
    //Defines corner points of the read image

    if(square)
    {
        std::cout << "square" << std::endl;
        for(int i = 0; i<4;i++){
            int random_point = 0;
            if(i == 0){
                random_point = center-0.4*s_int;
                int count = 3;
                while(image->data()[random_point]<colorThreshold_){
                    random_point = center-image->s()*0.4+count;
                    count+=3;
                    if(random_point > center)
                    {
                        random_point = center-0.4*s_int;
                        printf("Nothing found in negative x-direction!\n");
                        break;
                    }
                }
            }
            if(i == 1){
                random_point = center+0.4*s_int;
                int count = 3;
                while(image->data()[random_point]<colorThreshold_)
                {
                    random_point = center+image->s()*0.4-count;
                    count+=3;
                    if(random_point < center)
                    {
                        random_point = center+0.4*s_int;
                        printf("Nothing found in positive x-direction!\n");
                        break;
                    }
                }
            }
            if(i == 2){
                random_point = 1.4*s_int;
                while(image->data()[random_point]<colorThreshold_){
                    random_point += image->s();
                    if(random_point > center)
                    {
                        random_point = 1.4*s_int;
                        printf("Nothing found in positive y-direction!\n");
                        break;
                    }
                }
            }
            if(i == 3){
                random_point = image->getTotalDataSize()-image->s()*1.4;
                while(image->data()[random_point]<colorThreshold_){
                    random_point -= image->s();
                    if(random_point > center)
                    {
                        random_point = image->getTotalDataSize()-image->s()*1.4;
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

        delta_1 = corner_points[1]-corner_points[0];
        delta_2 = (corner_points[2]-corner_points[0])/t_int;

    /***********************************************************************************************/
    //Based on the amount of samplings, specific points are chosen from the relevant area

        if(samplingPoints < 5*image->t())
            samplingPoints = 5*image->t();                          //This is needed to get at least minimum accuracy

        int ppl = samplingPoints/(delta_2);
        int steppingPerLine = delta_1/(ppl-2+1);

        std::cout << ppl << " " <<steppingPerLine << std::endl;

        for(int i = 0; i<delta_2; i++){

            for(int j = 0; j < ppl;j++){
                if(j == ppl-1)
                    points_[count] = corner_points[0]+delta_2+s_int*i;
                else
                {
                    if(image->data()[corner_points[0]+steppingPerLine*j+s_int*i] == 0)
                            continue;
                    else
                    {
                        points_[count] = corner_points[0]+steppingPerLine*j+s_int*i;
                        count++;
                    }
                }
            }
        }

        samplingPoints = count;
        std::cout << count << std::endl;

        count = 0;

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

        count = 0;

    }

    int index = 0;


    /***********************************************************************************************/
    //Angles for later purpose are defined here

    int deltaWidth = 0;
    int deltaHeight = 0;
    float angleToNozzle = 0;
    float angleToPosition = 0;
    float particleFlow = 0;
    float particleRadius = 0;


    iBuf.centerX = center_width;
    iBuf.centerY = center_height;
    iBuf.samplingPoints = samplingPoints;
    iBuf.dataBuffer = new float[samplingPoints*6];

    for(count; count < samplingPoints; count++){

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

        //printf("%i %i\n", index, image->getTotalDataSize());
        particleFlow = 0.25*(  image->data()[index]+
                               image->data()[index+3]+
                image->data()[index+s_int]+
                image->data()[index+s_int+3]
                )*pixel_to_flow_;
        //Flow

        particleRadius = image->data()[index]*pixel_to_radius_;
        //Radius

        iBuf.dataBuffer[count*6] = angleToNozzle;
        iBuf.dataBuffer[count*6+1] = angleToPosition;
        iBuf.dataBuffer[count*6+2] = deltaWidth;
        iBuf.dataBuffer[count*6+3] = deltaHeight;
        iBuf.dataBuffer[count*6+4] = particleFlow;
        iBuf.dataBuffer[count*6+5] = particleRadius;

        //mystream << angleToNozzle << " " << angleToPosition << " " << deltaWidth << " " << deltaHeight << " " << particleFlow << " " << particleRadius << std::endl;

        //        frequency_[count] = 3*intensity_[count]/(4*pow(radius_[count],3)*Pi);

        //        min_particles += frequency_[count];




    };

    //mystream.close();


    std::cout << "Reading image completed!\n" << std::endl;

    return true;
}
