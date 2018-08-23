/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gen.h"

int randMax = RAND_MAX;

inline int sgn(float x)
{
    if(x == 0)
        return 0;
    else
        return (x>0) ? 1 : -1;
}

gen::gen(float pInit, class nozzle* owner)
{
    initPressure_ = pInit;
    owner_ = owner;

    //Set variables received from parser
    particleCount_ = parser::instance()->getReqParticles();
    nu = parser::instance()->getNu();
    densityOfFluid = parser::instance()->getDensityOfFluid();
    densityOfParticle = parser::instance()->getDensityOfParticle();
    reynoldsThreshold = parser::instance()->getReynoldsThreshold();
    cwTurb = parser::instance()->getCwTurb();
    cwModelType = parser::instance()->getCwModelType();
    iterations = parser::instance()->getIterations();
    minimum = parser::instance()->getMinimum();
    deviation = parser::instance()->getDeviation();

    //Basetransform - currently not needed
    transform_ = new osg::MatrixTransform;

    float t[] = {1,0,0,0,
                 0,1,0,0,
                 0,0,1,0,
                 0,0,0,1
                };
    osg::Matrix baseTransform;
    baseTransform.set(t);
    transform_->setMatrix(baseTransform);

    cover->getObjectsRoot()->addChild(transform_.get());

    geode_ = new osg::Geode;
    geode_->setName(owner->getName()+"Gen");
    transform_->addChild(geode_);

    coSphere_ = new coSphere();

}

gen::~gen()
{
    pVec.erase(pVec.begin(), pVec.end());
    cover->getObjectsRoot()->removeChild(transform_.get());
    geode_->removeDrawable(coSphere_);
}

void gen::init()
{
    prevHitDis = new float[particleCount_];
    prevHitDisCounter = new int[particleCount_];
    for(int i = 0; i< particleCount_;i++){
        prevHitDis[i] = 0;
        prevHitDisCounter[i] = 0;
    }
}

float gen::reynoldsNr(float v, double d)
{
    double reynolds_ = v*d*densityOfFluid/nu;
    if(reynolds_ >= reynoldsThreshold)
        return cwTurb;
    else
    {
        if(cwModelType.compare("STOKES") == 0)
        {
            cwLam = 24/reynolds_;
            return cwLam;
        }
        if(cwModelType.compare("MOLERUS") == 0)
        {
            cwLam = 24/reynolds_ + 0.4/sqrt(reynolds_)+0.4;
            return cwLam;
        }
        if(cwModelType.compare("MUSCHELK") == 0)
        {
            cwLam = 21.5/reynolds_ + 6.5/sqrt(reynolds_)+0.23;
            return cwLam;
        }

        if(cwModelType.compare("NONE") == 0)
        {
            cwLam = 0;
            return cwLam;
        }
        return 0.45;
    }
}

void gen::setCoSphere(osg::Vec3Array* pos)
{

    float* rVis = new float[particleCount_];
    for(int i = 0; i<particleCount_;i++)rVis[i] = pVec[i]->r*parser::instance()->getScaleFactor();
    coSphere_->setMaxRadius(100);

    if(parser::instance()->getIsAMD() == 1)
        coSphere_->setRenderMethod(coSphere::RENDER_METHOD_CG_SHADER);
    else
        coSphere_->setRenderMethod(coSphere::RENDER_METHOD_ARB_POINT_SPRITES);    //Doesn't work properly on AMD RADEON 7600M

    //Set particle parameters with radius for visualisation
    coSphere_->setCoords(particleCount_,
                            pos,
                            rVis);
    geode_->addDrawable(coSphere_);
}

void gen::updateCoSphere(){
    for(int i = 0; i< particleCount_;i++)
        coSphere_->updateCoords(i, pVec[i]->pos);

    if(outOfBoundCounter >= 0.65*particleCount_)
    {
        outOfBound = true;        
    }
    else
    {
        outOfBoundCounter = 0;
    }
}

void gen::updatePos(osg::Vec3 boundingBox){  

    float floorHeight = VRSceneGraph::instance()->floorHeight()*0.001;
    tCur = parser::instance()->getRendertime()/60;
    float timesteps = tCur/iterations;

    for(int i = 0; i<particleCount_;i++){

        particle* p = pVec[i];

        if(p->particleOutOfBound)
        {
            outOfBoundCounter++;
            continue;
        }

        osg::Vec3 velMed = osg::Vec3(0,0,0);                                    //median speed over iterations

        for(int it = 0; it<iterations; it++)
        {

            float v = p->velocity.length();                                     //get absolute velocity

            float cwTemp = reynoldsNr(v, 2*p->r);

            p->pos += p->velocity*timesteps;                                      //set new positions

#if 1
            //reynolds

            float k = 0.5*densityOfFluid*p->r*p->r*Pi*cwTemp/p->m;              //constant value for wind force

//            vx[i] -= timesteps*(k*v*vx[i]+gravity.x())/2;
//            vy[i] -= timesteps*(k*v*vy[i]+gravity.y())/2;
//            vz[i] -= timesteps*(k*v*vz[i]+gravity.z())/2;

            p->velocity -= p->velocity*k*v*timesteps*0.5+gravity*timesteps/2;                //new velocity

            velMed += p->velocity;                                              //sum of velocity over iterations

        }

        velMed /= iterations;                                                   //median velocity


#endif

#if 0
        //stokes

        vx[i] -= tCur/m[i]*(6*Pi*nu*vx[i]*r[i]+gravity.x()*m[i]);

        vy[i] -= tCur/m[i]*(6*Pi*nu*vy[i]*r[i]+gravity.y()*m[i]);

        vz[i] -= tCur/m[i]*(6*Pi*nu*vz[i]*r[i]+gravity.z()*m[i]);

#endif

        if(p->pos.z()<floorHeight){
            if(p->firstHit == true)
            {
                p->particleOutOfBound = true;
                outOfBoundCounter++;
            }
            else
            {
                if((float)rand()/(float)randMax>0.5)
                {
                p->velocity.x() *= ((float)rand()/randMax-0.5)*0.5;
                p->velocity.y() *= ((float)rand()/randMax-0.5)*0.5;
                }
                p->firstHit = true;
            }
        }
        if(p->pos.x() > boundingBox.x() || p->pos.y() > boundingBox.y() || p->pos.z() > boundingBox.z() ||
               p->pos.x()<(-boundingBox.x()) || p->pos.y() < (-boundingBox.y()) )
        {
            p->particleOutOfBound = true;
            outOfBoundCounter++;
        }

        particle rayP = *p;
        rayP.velocity = velMed*tCur;

        rayP = raytracer::instance()->handleParticleData(rayP);

        if(rayP.hit != 0)
        {
            prevHitDis[i] = rayP.pos.z();
            prevHitDisCounter[i] = 0;

            if(abs(p->velocity.y())+abs(p->pos.y()) > abs(rayP.pos.z()))
            {
                p->particleOutOfBound = true;
                coSphere_->setColor(i,0,0,1,1);
            }
        }

    }

    updateCoSphere();
}







imageGen::imageGen(pImageBuffer* iBuf, float pInit, class nozzle* owner):gen(pInit, owner){
    iBuf_ = iBuf;
    particleCount_ = iBuf->samplingPoints;
}

imageGen::~imageGen(){

}

void imageGen::seed(){
    osg::Vec3Array* pos = new osg::Vec3Array;
    for(int i = 0; i < iBuf_->samplingPoints; i++)
    {
        //for(int j = 0; j<p->frequency_[i]; j++){
        //float winkel = Winkel*Pi/180;
        particle* p = new particle();
        osg::Vec3 spitze = osg::Vec3f(0,1,0);
        osg::Matrix spray_pos = owner_->getMatrix();
        osg::Vec3 duese = spray_pos.getTrans();
        spitze = spray_pos*spitze;

        float offset = 0.001;                                                           //Needed for rotation of the nozzle (1mm)

        p->pos.x() = duese.x()+spitze.x()*offset;
        p->pos.y() = duese.y()+spitze.y()*offset;
        p->pos.z() = duese.z()+spitze.z()*offset;
        p->r = iBuf_->dataBuffer[i*6+5]*0.001;
        p->m = 4 / 3 * p->r * p->r * p->r * Pi * densityOfParticle;

        float v = sqrt(2*initPressure_*100000/densityOfParticle);                           //Initial speed of particle

        //v = 4;
        float hypotenuse = sqrt(pow(iBuf_->dataBuffer[i*6+2],2)+pow(iBuf_->dataBuffer[i*6+3],2));
        float d_angle = atan(iBuf_->dataBuffer[i*6+3]/iBuf_->dataBuffer[i*6+2]);

        p->velocity.x() = v*sin(iBuf_->dataBuffer[i*6])*cos(iBuf_->dataBuffer[i*6+1]);
        p->velocity.y() = v*cos(iBuf_->dataBuffer[i*6]);
        p->velocity.z() = v*sin(iBuf_->dataBuffer[i*6])*sin(iBuf_->dataBuffer[i*6+1]);


        osg::Quat a = spray_pos.getRotate();
        osg::Matrix spray_rot;
        spray_rot.setRotate(a);

        p->velocity = spray_rot*p->velocity;

        pVec.push_back(p);
        pos->push_back(p->pos);
    }
    setCoSphere(pos);
}








standardGen::standardGen(float sprayAngle, std::string decoy, float pInit, class nozzle* owner):gen(pInit, owner){
    sprayAngle_ = sprayAngle;
    decoy_ = decoy;
}

void standardGen::seed(){

    float redDegree = 0;
    std::string type = "NONE";

    if(decoy_.compare("NONE") == 0)
    {
        type = "NONE";
    }
    else
    {
        std::string line;
        std::stringstream ssLine(decoy_);
        std::getline(ssLine,line,'_');

        if(line.compare("RING") == 0)
        {
            type = "RING";
            std::getline(ssLine,line,'\n');
            try
            {
                redDegree = stof(line);
            }//try

            catch(const std::invalid_argument& ia)
            {
                std::cerr << "Invalid argument: " << ia.what() << std::endl;
            }
        }

        if(line.compare("BAR") == 0)
        {
            type = "BAR";
            std::getline(ssLine,line,'\n');
            try
            {
                redDegree = stof(line);
            }//try

            catch(const std::invalid_argument& ia)
            {
                std::cerr << "Invalid argument: " << ia.what() << std::endl;
            }
        }
    }

    osg::Vec3Array* pos = new osg::Vec3Array;

    for(int i = 0; i< particleCount_; i++){

        particle* p = new particle();
        osg::Vec3 spitze = osg::Vec3f(0,1,0);
        osg::Matrix spray_pos = owner_->getMatrix();
        osg::Vec3 duese = spray_pos.getTrans();
        spitze = spray_pos*spitze;

        float randAngle = ((float)rand())/(float)randMax;

        float offset = 0.001;                                                               //Needed for rotation of the nozzle (1mm)
        float sprayAngle = 0;
        if(redDegree != 0 && type.compare("RING") == 0)
            sprayAngle = (sprayAngle_-redDegree+redDegree*randAngle)*Pi/180*0.5;
        else
            sprayAngle = -sprayAngle_*0.5*Pi/180+sprayAngle_*randAngle*Pi/180;
        float massRand = ((float)rand())/(float)randMax;									//random value between 0 and 1

        p->pos.x() = duese.x()+spitze.x()*offset;
        p->pos.y() = duese.y()+spitze.y()*offset;
        p->pos.z() = duese.z()+spitze.z()*offset;

        p->r = (getMinimum()+getDeviation()*massRand)*0.5;

        p->m = 4 / 3 * p->r * p->r * p->r * Pi * densityOfParticle;

        float v = sqrt(2*initPressure_*100000/densityOfParticle);                            //Initial speed of particle
        float d_angle = 0;
        if(type.compare("BAR") == 0)
        {
            d_angle = -redDegree*Pi/180*0.5+(float)rand()/(float)randMax*redDegree*Pi/180;                    //BAR is only a rectangle
            p->velocity.x() = v*sin(sprayAngle);
            p->velocity.y() = v*cos(sprayAngle);
            p->velocity.z() = v/**sin(sprayAngle)*/*sin(d_angle);
        }
        else
        {
            d_angle = (float)rand()/(float)randMax*2*Pi;                                //NONE and RING are still radially symmetric
            p->velocity.x() = v*sin(sprayAngle)*cos(d_angle);
            p->velocity.y() = v*cos(sprayAngle);
            p->velocity.z() = v*sin(sprayAngle)*sin(d_angle);
        }





        osg::Quat a = spray_pos.getRotate();
        osg::Matrix spray_rot;
        spray_rot.setRotate(a);

        p->velocity = spray_rot*p->velocity;
        pVec.push_back(p);
        pos->push_back(p->pos);

    }
    setCoSphere(pos);
}
