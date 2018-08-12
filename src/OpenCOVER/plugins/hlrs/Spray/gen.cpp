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
    delete x;
    delete y;
    delete z;
    delete vx;
    delete vy;
    delete vz;
    delete m;
    delete r;
    delete firstHit;
    delete particleOutOfBound;
    cover->getObjectsRoot()->removeChild(transform_.get());
    geode_->removeDrawable(coSphere_);
}

void gen::init()
{
    x = new float[particleCount_];
    y = new float[particleCount_];
    z = new float[particleCount_];
    vx = new float[particleCount_];
    vy = new float[particleCount_];
    vz = new float[particleCount_];
    r = new float[particleCount_];
    m = new float[particleCount_];

    particleOutOfBound = new bool[particleCount_];
    firstHit = new bool[particleCount_];

    prevHitDis = new float[particleCount_];
    prevHitDisCounter = new int[particleCount_];
    for(int i = 0; i< particleCount_;i++){
        firstHit[i] = false;
        particleOutOfBound[i] = false;
        prevHitDis[i] = 0;
        prevHitDisCounter[i] = 0;
    }
}

float gen::reynoldsNr(float v, float d)
{
    float reynolds_ = v*d*densityOfFluid/nu;
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
        return 0.45;
    }
}

void gen::setCoSphere()
{
    float* rVis = new float[particleCount_];
    for(int i = 0; i<particleCount_;i++)rVis[i] = r[i]*parser::instance()->getScaleFactor();
    coSphere_->setMaxRadius(100);

    if(parser::instance()->getIsAMD() == 1)
        coSphere_->setRenderMethod(coSphere::RENDER_METHOD_CG_SHADER);
    else
        coSphere_->setRenderMethod(coSphere::RENDER_METHOD_ARB_POINT_SPRITES);    //Doesn't work properly on AMD RADEON 7600M

    //coSphere_->setRenderMethod(coSphere::RENDER_METHOD_CG_SHADER);

    //Set particle parameters with radius for visualisation
    coSphere_->setCoords(particleCount_,
                            x,
                            y,
                            z,
                            rVis);
    geode_->addDrawable(coSphere_);
}

void gen::updateCoSphere(){
    coSphere_->updateCoords(x, y, z);
    t += 0.001;

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
    tCur = parser::instance()->getRendertime();
    for(int i = 0; i<particleCount_;i++){

        if(particleOutOfBound[i])
        {
            outOfBoundCounter++;
            continue;
        }

        osg::Quat a = owner_->getMatrix().getRotate();
        osg::Matrix spray_rot;
        spray_rot.setRotate(a);
        gravity = spray_rot*osg::Vec3f(0,0,g);

        float cwTemp = reynoldsNr(sqrt(pow(vx[i],2)+pow(vy[i],2)+pow(vz[i],2)), 2*r[i]);

        x[i] += vx[i];
        y[i] += vy[i];
        z[i] += vz[i];

#if 1
        //reynolds

//        float vTemp = tCur/m[i]*(sgn(vx[i])*0.5*cwTemp*vx[i]*vx[i]*1.18*pow(r[i],2)*Pi+gravity.x()*m[i]);
//        if(abs(vTemp)>abs(vx[i]))
//            vx[i] = 0;
//        else
//            vx[i] -= vTemp;

//        vTemp = tCur/m[i]*(sgn(vy[i])*0.5*cwTemp*vy[i]*vy[i]*1.18*pow(r[i],2)*Pi+gravity.y()*m[i]);
//        if(abs(vTemp)>abs(vy[i]))
//            vy[i] = 0;
//        else
//            vy[i] -= vTemp;

//        vTemp = tCur/m[i]*(sgn(vz[i])*0.5*cwTemp*vz[i]*vz[i]*1.18*pow(r[i],2)*Pi+gravity.z()*m[i]);
//        if(abs(vTemp)>abs(vz[i]))
//            vz[i] = 0;
//        else
//            vz[i] -= vTemp;


//        vx[i] -= tCur/m[i]*(sgn(vx[i])*0.5*cwTemp*vx[i]*vx[i]*densityOfFluid*pow(r[i],2)*Pi+gravity.x()*m[i]);
//        vy[i] -= tCur/m[i]*(sgn(vy[i])*0.5*cwTemp*vy[i]*vy[i]*densityOfFluid*pow(r[i],2)*Pi+gravity.y()*m[i]);
//        vz[i] -= tCur/m[i]*(sgn(vz[i])*0.5*cwTemp*vz[i]*vz[i]*densityOfFluid*pow(r[i],2)*Pi+gravity.z()*m[i]);

        //Differentialansatz mit Reynolds
        vx[i] = 1/(0.5*densityOfFluid*pow(r[i],2)*Pi*cwTemp/m[i]*tCur+1/vx[i])-gravity.x()*tCur;
        vy[i] = 1/(0.5*densityOfFluid*pow(r[i],2)*Pi*cwTemp/m[i]*tCur+1/vy[i])-gravity.y()*tCur;
        vz[i] = 1/(0.5*densityOfFluid*pow(r[i],2)*Pi*cwTemp/m[i]*tCur+1/vz[i])-gravity.z()*tCur;

        if(i == 20) printf("%f %f %f %i\n", vx[i], vy[i], vz[i], i);
        if(i == 40) printf("%f %f %f %i\n", vx[i], vy[i], vz[i], i);
        if(i == 60) printf("%f %f %f %i\n", vx[i], vy[i], vz[i], i);
        if(i == 80) printf("%f %f %f %i\n", vx[i], vy[i], vz[i], i);
#endif

#if 0
        //stokes

        vx[i] -= tCur/m[i]*(6*Pi*nu*vx[i]*r[i]+gravity.x()*m[i]);

        vy[i] -= tCur/m[i]*(6*Pi*nu*vy[i]*r[i]+gravity.y()*m[i]);

        vz[i] -= tCur/m[i]*(6*Pi*nu*vz[i]*r[i]+gravity.z()*m[i]);

#endif

        if(y[i]<floorHeight){
            if(firstHit[i] == true)
            {
                particleOutOfBound[i] = true;
                outOfBoundCounter++;
            }
            else
            {
                if((float)rand()/(float)randMax>0.5)
                {
                vx[i] = vx[i]*((float)rand()/randMax-0.5)*0.5;
                vy[i] = vy[i]*((float)rand()/randMax-0.5)*0.5;
                }
                firstHit[i] = true;
            }
        }
        if(x[i] > boundingBox.x() || y[i] > boundingBox.y() || z[i] > boundingBox.z() ||
                x[i]<(-boundingBox.x()) || z[i] < (-boundingBox.z()) )
        {
            particleOutOfBound[i] = true;
            outOfBoundCounter++;
        }


        p.x = x[i];
        p.y = y[i];
        p.z = z[i];
        p.vx = vx[i]*tCur;
        p.vy = vy[i]*tCur;
        p.vz = vz[i]*tCur;
        p.hit = 0;

        p = raytracer::instance()->handleParticleData(p);

        if(p.hit != 0)
        {
            prevHitDis[i] = p.z;
            prevHitDisCounter[i] = 0;

            if(abs(y[i])+abs(vy[i]) > abs(p.z))
            {
                particleOutOfBound[i] = true;
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
    for(int i = 0; i < iBuf_->samplingPoints; i++)
    {
        //for(int j = 0; j<p->frequency_[i]; j++){
        //float winkel = Winkel*Pi/180;

        osg::Vec3f spitze = osg::Vec3f(0,1,0);
        osg::Matrix spray_pos = owner_->getMatrix();
        osg::Vec3 duese = spray_pos.getTrans();
        spitze = spray_pos*spitze;

        float offset = 0;                                                 //Abstand Duesenmitte zur emittierenden Seite

        x[i] = duese.x()+spitze.x()*offset;
        y[i] = duese.y()+spitze.y()*offset;
        z[i] = duese.z()+spitze.z()*offset;
        r[i] = iBuf_->dataBuffer[i*6+5]*0.001;
        m[i] = 4/3*pow(r[i],3)*Pi*1000;

        //F = p*A; F = m*a
        //p*A = m*a = m*dv/t
        //v = p*A/m*t; t = 1s
        float v = (getInitPressure()*0.1*pow(r[i],2)*Pi)/(m[i]);                           //Geschwindigkeitsgradient des Partikels
        //v = 4;
        float hypotenuse = sqrt(pow(iBuf_->dataBuffer[i*6+2],2)+pow(iBuf_->dataBuffer[i*6+3],2));
        float d_angle = atan(iBuf_->dataBuffer[i*6+3]/iBuf_->dataBuffer[i*6+2]);

        vx[i] = v*sin(iBuf_->dataBuffer[i*6])*cos(iBuf_->dataBuffer[i*6+1]);
        vz[i] = v*sin(iBuf_->dataBuffer[i*6])*sin(iBuf_->dataBuffer[i*6+1]);
        vy[i] = v*cos(iBuf_->dataBuffer[i*6]);

        osg::Vec3f buffer = osg::Vec3f(vx[i],vy[i],vz[i]);

        osg::Quat a = spray_pos.getRotate();
        osg::Matrix spray_rot;
        spray_rot.setRotate(a);

        buffer = spray_rot*buffer;

        vx[i] = buffer.x();
        vy[i] = buffer.y();
        vz[i] = buffer.z();
    }
    setCoSphere();
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


    for(int i = 0; i< particleCount_; i++){

        osg::Vec3f spitze = osg::Vec3f(0,1,0);
        osg::Matrix spray_pos = owner_->getMatrix();
        osg::Vec3 duese = spray_pos.getTrans();
        spitze = spray_pos*spitze;

        float randAngle = ((float)rand())/(float)randMax;

        float offset = 0.001;                                                 //Abstand Duesenmitte zur emittierenden Seite
        float sprayAngle = 0;
        if(redDegree != 0 && type.compare("RING") == 0)
            sprayAngle = (sprayAngle_-redDegree+redDegree*randAngle)*Pi/180*0.5;
        else
            sprayAngle = sprayAngle_*randAngle*Pi/180*0.5;
        float massRand = ((float)rand())/(float)randMax;											  //random value between 0 and 1

        x[i] = duese.x()+spitze.x()*offset;
        y[i] = duese.y()+spitze.y()*offset;
        z[i] = duese.z()+spitze.z()*offset;

        r[i] = (getMinimum()+getDeviation()*massRand)*0.5;
        m[i] = 4/3*pow(r[i],3)*Pi*densityOfParticle;

        float v = getInitPressure()*0.1*pow(r[i],2)*Pi/m[i];                            //Beschleunigung des Partikels
        float d_angle = 0;
        if(type.compare("BAR") == 0)
        {
            d_angle = (float)rand()/(float)randMax*redDegree*Pi/180;                    //BAR is only a rectangle
        }
        else
            d_angle = (float)rand()/(float)randMax*2*Pi;                                //NONE and RING are still radially symmetric



        vx[i] = v*sin(sprayAngle)*cos(d_angle);
        vz[i] = v*sin(sprayAngle)*sin(d_angle);
        vy[i] = v*cos(sprayAngle);

        osg::Vec3f buffer = osg::Vec3f(vx[i],vy[i],vz[i]);

        osg::Quat a = spray_pos.getRotate();
        osg::Matrix spray_rot;
        spray_rot.setRotate(a);

        buffer = spray_rot*buffer;

        vx[i] = buffer.x();
        vy[i] = buffer.y();
        vz[i] = buffer.z();

    }
    setCoSphere();
}
