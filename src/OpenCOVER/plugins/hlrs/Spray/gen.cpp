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

inline float absf(float x)
{
    if(x == 0)
        return 0;
    if(x > 0)
        return x;
    else
        return x*(-1);
}

float gen::gaussian(float value)
{
    return 1/(sqrt(Pi)*alpha*gaussamp)*exp((-1)*value*value/(alpha*alpha));
}

//float gen::gaussian(float value)
//{
//    return 1/(sqrt(2*Pi)*alpha*gaussamp)*exp((-1)*(value*value-deviation)/(alpha*alpha));
//}

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
    reynoldsLimit = parser::instance()->getReynoldsLimit();
    cwTurb = parser::instance()->getCwTurb();
    iterations = parser::instance()->getIterations();
    minimum = parser::instance()->getMinimum();
    deviation = parser::instance()->getDeviation();
    alpha = parser::instance()->getAlpha();
    gaussamp = gaussian(0);

    if(parser::instance()->getCwModelType().compare("STOKES") == 0)
        cwModelType = CW_STOKES;
    if(parser::instance()->getCwModelType().compare("MOLERUS") == 0)
        cwModelType = CW_MOLERUS;
    if(parser::instance()->getCwModelType().compare("MUSCHELK") == 0)
        cwModelType = CW_MUSCHELK;
    if(parser::instance()->getCwModelType().compare("NONE") == 0)
        cwModelType = CW_NONE;

    geode_ = new osg::Geode;
    geode_->setName("Gen"+owner->getName());
    cover->getObjectsRoot()->addChild(geode_);

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
    //additional parameters can be initialized here
}

inline float gen::reynoldsNr(float v, double d)
{
    double reynolds_ = v*d*densityOfFluid/nu;
    if(reynolds_ >= reynoldsThreshold)
    {
        if(reynolds_ > reynoldsLimit)
            printf("Drag modelling behaves correctly till Re = %i! Currently Re = %f\n Propagation may be incorrect!\n",reynoldsLimit, reynolds_);
        return cwTurb;
    }
    else
    {
        if(cwModelType == CW_STOKES)
        {
            cwLam = 24/reynolds_;
            return cwLam > cwTurb ? cwLam : cwTurb;
            //return cwLam;
        }
        if(cwModelType == CW_MOLERUS)
        {
            cwLam = 24/reynolds_ + 0.4/sqrt(reynolds_)+0.4;
            return cwLam;
        }
        if(cwModelType == CW_MUSCHELK)
        {
            cwLam = 21.5/reynolds_ + 6.5/sqrt(reynolds_)+0.23;
            return cwLam;
        }

        if(cwModelType == CW_NONE)
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
    for(int i = 0; i<particleCount_;i++)
        rVis[i] = pVec[i]->r*parser::instance()->getScaleFactor();

    if(parser::instance()->getSphereRenderType() == 1)
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

    if(outOfBoundCounter >= removeCount*particleCount_)
    {
        outOfBound = true;
    }
    else
    {
        outOfBoundCounter = 0;
    }
}

void gen::updatePos(osg::Vec3 boundingBox)
{
    tCur = parser::instance()->getRendertime()/60;
    float timesteps = tCur/iterations;

    //    float sumElapsedTime = 0.0;
    //    std::clock_t begin;
    //    std::clock_t end;

    for(int it = 0; it<iterations; it++)
    {
        outOfBoundCounter = 0;
        for(int i = 0; i<particleCount_;i++)
        {

            particle* p = pVec[i];

            if(p->particleOutOfBound)
            {
                outOfBoundCounter++;
                continue;
            }

            //            begin = clock();
            float elapsedTime = raytracer::instance()->checkForHit(*p, timesteps);
            //            end = clock();
            //            sumElapsedTime += double(end - begin) / CLOCKS_PER_SEC;

            if(elapsedTime >= 0)                                                       //hit was registered by embree
            {
                p->particleOutOfBound = true;                                               //particle has hit an object
                p->pos += p->velocity*timesteps*elapsedTime;
                coSphere_->setColor(i,0,0,1,1);
            }

            else
            {
                float v = p->velocity.length();                                     //get absolute velocity

                float cwTemp = reynoldsNr(v, 2*p->r);

                p->pos += p->velocity*timesteps;                                    //set new positions

                float k = 0.5*densityOfFluid*p->r*p->r*Pi*cwTemp/p->m;              //constant value for wind force

                p->velocity -= p->velocity*k*v*timesteps*0.5+gravity*timesteps/2;   //new velocity

                if(p->pos.z()<(-boundingBox.z()))
                {
                    if(p->firstHit == true)
                    {
                        p->particleOutOfBound = true;
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
                }
            }   //else

        }
    }
    //    printf("elapsed time for RT all %f\n", sumElapsedTime);
    updateCoSphere();
}

void gen::updateAll(osg::Vec3 boundingBox)
{
    tCur = parser::instance()->getRendertime()/60;

//    std::clock_t begin = clock();
    raytracer::instance()->checkAllHits(pVec, tCur);
//    std::clock_t end = clock();
//    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    printf("elapsed time for RT all %f\n", elapsed_secs);


    for(int i = 0; i<particleCount_;i++)
    {
        particle* p = pVec[i];

        if(p->particleOutOfBound)
        {
            outOfBoundCounter++;
            continue;
        }

        if(p->time >= 0)                                                       //hit was registered by embree
        {
            p->particleOutOfBound = true;                                               //particle has hit an object
            p->pos += p->velocity*tCur*p->time;
            coSphere_->setColor(i,0,0,1,1);
        }

        else
        {
            float v = p->velocity.length();                                     //get absolute velocity

            float cwTemp = reynoldsNr(v, 2*p->r);

            p->pos += p->velocity*tCur;                                    //set new positions

            float k = 0.5*densityOfFluid*p->r*p->r*Pi*cwTemp/p->m;              //constant value for wind force

            p->velocity -= p->velocity*k*v*tCur*0.5+gravity*tCur/2;   //new velocity

            if(p->pos.z()<(-boundingBox.z()))
            {
                if(p->firstHit == true)
                {
                    p->particleOutOfBound = true;
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
            }
        }   //else


    }
    updateCoSphere();

}




imageGen::imageGen(pImageBuffer* iBuf, float pInit, class nozzle* owner):gen(pInit, owner){
    iBuf_ = iBuf;
    particleCount_ = iBuf->samplingPoints;
}

void imageGen::seed(){
    int newParticleCount = 0;                                                               //Evaluated pixel != emitted pixels
    osg::Vec3Array* pos = new osg::Vec3Array;
    for(int i = 0; i < iBuf_->samplingPoints; i++)
    {
        for(int j = 0; j < (int)iBuf_->dataBuffer[i*5+4]+1; j++){
            particle* p = new particle();
            osg::Vec3 spitze = osg::Vec3f(0,1,0);
            osg::Matrix sprayPos = owner_->getMatrix().inverse(owner_->getMatrix());
            osg::Vec3 duese = owner_->getMatrix().getTrans();
            spitze = spitze*sprayPos;

            float offset = 0.001;                                                           //Needed for rotation of the nozzle (1mm)

            float massRand = ((float)rand())/(float)randMax;

            p->pos.x() = duese.x()+spitze.x()*offset;
            p->pos.y() = duese.y()+spitze.y()*offset;
            p->pos.z() = duese.z()+spitze.z()*offset;
            p->r = (getMinimum()+getDeviation()*gaussian(massRand))*0.5;
            p->m = 4 / 3 * p->r * p->r * p->r * Pi * densityOfParticle;
            float xdist = cos(2*Pi/iBuf_->dataBuffer[i*5+4]*j)*0.01;                       //Considers distribution around center
            float ydist = sin(2*Pi/iBuf_->dataBuffer[i*5+4]*j)*0.01;                       //Otherwise all particles would travel the same trajectory

            float v = sqrt(2*initPressure_*100000/densityOfParticle);                           //Initial speed of particle

            p->velocity.x() = v*sin(iBuf_->dataBuffer[i*5])*cos(iBuf_->dataBuffer[i*5+1]) + xdist;
            p->velocity.y() = v*cos(iBuf_->dataBuffer[i*5]);
            p->velocity.z() = v*sin(iBuf_->dataBuffer[i*5])*sin(iBuf_->dataBuffer[i*5+1]) + ydist;


            sprayPos.setTrans(0,0,0);

            p->velocity = sprayPos*p->velocity;
            pVec.push_back(p);
            pos->push_back(p->pos);
            newParticleCount++;
        }

    }
    particleCount_ = newParticleCount;
    setCoSphere(pos);
    pos->unref();
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
        osg::Vec3 spitze = osg::Vec3f(0,-1,0);
        osg::Matrix sprayPos = owner_->getMatrix().inverse(owner_->getMatrix());

        osg::Vec3 duese = owner_->getMatrix().getTrans();
        spitze = spitze*sprayPos;

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

        p->r = (getMinimum()+getDeviation()*gaussian(massRand))*0.5;

        p->m = 4 / 3 * p->r * p->r * p->r * Pi * densityOfParticle;

        float v = sqrt(2*initPressure_*100000/densityOfParticle);                            //Initial speed of particle
        float d_angle = 0;
        if(type.compare("BAR") == 0)
        {
            d_angle = -redDegree*Pi/180*0.5+(float)rand()/(float)randMax*redDegree*Pi/180;                    //BAR is only a rectangle
            p->velocity.x() = v*sin(sprayAngle);
            p->velocity.y() = v*cos(sprayAngle);
            p->velocity.z() = v*sin(d_angle);
        }
        else
        {
            d_angle = (float)rand()/(float)randMax*2*Pi;                                //NONE and RING are still radially symmetric
            p->velocity.x() = v*sin(sprayAngle)*cos(d_angle);
            p->velocity.y() = v*cos(sprayAngle);
            p->velocity.z() = v*sin(sprayAngle)*sin(d_angle);
        }

        sprayPos.setTrans(0,0,0);

        p->velocity = sprayPos*p->velocity;
        pVec.push_back(p);
        pos->push_back(p->pos);

    }
    setCoSphere(pos);
    pos->unref();
}
