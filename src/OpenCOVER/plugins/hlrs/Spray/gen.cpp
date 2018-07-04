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

    particleCount_ = parser::instance()->getReqParticles();
    nu = parser::instance()->getNu();
    densityOfFluid = parser::instance()->getDensityOfFluid();
    reynoldsThreshold = parser::instance()->getReynoldsThreshold();
    cwTurb = parser::instance()->getCwTurb();
    cwModelType = parser::instance()->getCwModelType();

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

    //geode_ = new osg::ref_ptr<osg::Geode>;
    geode_ = new osg::Geode;
    geode_->setName(owner->getName()+"Gen");
    transform_->addChild(geode_);

    coSphere_ = new coSphere();

    init();
}

gen::~gen(){
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

void gen::init(){
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
    for(int i = 0; i< particleCount_;i++){
        firstHit[i] = false;
        particleOutOfBound[i] = false;
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
        }
        if(cwModelType.compare("MOLERUS") == 0)
        {
            cwLam = 24/reynolds_ + 0.4/sqrt(reynolds_)+0.4;
        }
        if(cwModelType.compare("MUSCHELK") == 0)
        {
            cwLam = 21.5/reynolds_ + 6.5/sqrt(reynolds_)+0.23;
        }
        return 0.1;
    }
}

void gen::setCoSphere(){
    float* rVis = new float[particleCount_];
    for(int i = 0; i<particleCount_;i++)rVis[i] = r[i]*100000;
    coSphere_->setMaxRadius(100);
    coSphere_->setRenderMethod(coSphere::RENDER_METHOD_CPU_BILLBOARDS);
    coSphere_->setCoords(particleCount_,
                            x,
                            y,
                            z,
                            rVis);
    //std::cout << "Max Radius: " << coSphere_->getMaxRadius() <<std::endl;
    geode_->addDrawable(coSphere_);
}

void gen::updateCoSphere(){
    coSphere_->updateCoords(x, y, z);
    //t += cover->frameDuration();
    t += 0.001;
    if(outOfBoundCounter >= 0.65*particleCount_)
    {
        outOfBound = true;
        //std::cout << "Spheres hitting geometry: " << sphereHitCounter << std::endl;
    }
    else
    {
        outOfBoundCounter = 0;
    }
}

void gen::updatePos(osg::Vec3 boundingBox){  

    float floorHeight = VRSceneGraph::instance()->floorHeight();
    for(int i = 0; i<particleCount_;i++){

        if(particleOutOfBound[i])
        {
            outOfBoundCounter++;
            continue;
        }

        osg::Quat a = owner_->getMatrix().getRotate();
        osg::Matrix spray_rot;
        spray_rot.setRotate(a);
        gravity = osg::Vec3f(0,0,g);

        float cwTemp = reynoldsNr(sqrt(pow(vx[i],2)+pow(vy[i],2)+pow(vz[i],2)), 2*r[i]);

        x[i] += vx[i]*tCur*1000;
        y[i] += vy[i]*tCur*1000;
        z[i] += vz[i]*tCur*1000;

#if 1        //reynolds
        vx[i] -= tCur/m[i]*(sgn(vx[i])*0.5*cwTemp*vx[i]*vx[i]*1.18*pow(r[i],2)*Pi+gravity.x()*m[i]);
        //if(vx[i] < 0) vx[i] = 0;
        vy[i] -= tCur/m[i]*(sgn(vy[i])*0.5*cwTemp*vy[i]*vy[i]*1.18*pow(r[i],2)*Pi+gravity.y()*m[i]);
        //if(vy[i] < 0) vy[i] = 0;
        vz[i] -= tCur/m[i]*(sgn(vz[i])*0.5*cwTemp*vz[i]*vz[i]*1.18*pow(r[i],2)*Pi+gravity.z()*m[i]);
        //if(vzTemp/vz[i] < 0) vz[i] = 0;

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
                //sphereHitCounter++;
                particleOutOfBound[i] = true;
            }
            else
            {
                if((float)rand()/(float)randMax>0.5)
                {
                vx[i] = vx[i]*((float)rand()/randMax-0.5)*0.5;
                vy[i] = vy[i]*((float)rand()/randMax-0.5)*0.5;
                }
                firstHit[i] = true;
                //sphereHitCounter++;
            }
        }
        if(x[i] > boundingBox.x() || y[i] > boundingBox.y() || z[i] > boundingBox.z() ||
                x[i]<(-boundingBox.x()) || y[i] < (-boundingBox.y()) || z[i] < (-boundingBox.z()) )particleOutOfBound[i] = true;


        p.x = x[i];
        p.y = y[i];
        p.z = z[i];
        p.vx = vx[i]*0.001;
        p.vy = vy[i]*0.001;
        p.vz = vz[i]*0.001;
        p.hit = 0;

        p = raytracer::instance()->handleParticleData(p);

        if(p.hit != 0 && abs(p.z) < abs(y[i]))
        {
            particleOutOfBound[i] = true;
            coSphere_->setColor(i,0,0,1,1);

            //vx[i] = vy[i] = vz[i] = 0;
        }
    }

    updateCoSphere();
}







imageGen::imageGen(pImageBuffer* iBuf, float pInit, class nozzle* owner):gen(pInit, owner){
    iBuf_ = iBuf;
    particleCount_ = iBuf->samplingPoints;
    init();
    seed();
    setColor(getColor());
}

imageGen::~imageGen(){

}

void imageGen::seed(){
    for(int i = 0; i < iBuf_->samplingPoints; i++)
    {
        //for(int j = 0; j<p->frequency_[i]; j++){
        //float winkel = Winkel*Pi/180;

        osg::Vec3f spitze = osg::Vec3f(0,0,1);
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
        float v = (getInitPressure()*pow(r[i],2)*Pi)/(m[i]);                           //Geschwindigkeitsgradient des Partikels
        v = 4;
        float hypotenuse = sqrt(pow(iBuf_->dataBuffer[i*6+2],2)+pow(iBuf_->dataBuffer[i*6+3],2));
        float d_angle = atan(iBuf_->dataBuffer[i*6+3]/iBuf_->dataBuffer[i*6+2]);

        vx[i] = v*sin(iBuf_->dataBuffer[i*6])*cos(iBuf_->dataBuffer[i*6+1]);
        vz[i] = v*sin(iBuf_->dataBuffer[i*6])*sin(iBuf_->dataBuffer[i*6+1]);
        vy[i] = v*cos(iBuf_->dataBuffer[i*6]);

        osg::Vec3f buffer = osg::Vec3f(vx[i],vy[i],vz[i]);

        osg::Matrix spray_normal;
        spray_normal.orthoNormal(spray_pos);

        osg::Quat a = spray_pos.getRotate();
        osg::Matrix spray_rot;
        spray_rot.setRotate(a);

        buffer = spray_normal*spray_rot*buffer;

        vx[i] = buffer.x();
        vy[i] = buffer.y();
        vz[i] = buffer.z();
    }
    setCoSphere();
}








standardGen::standardGen(float sprayAngle, const char *decoy, float pInit, class nozzle* owner):gen(pInit, owner){
    sprayAngle_ = sprayAngle;
    decoy_ = decoy;

    seed();
    setColor(getColor());
}

void standardGen::seed(){

    for(int i = 0; i< particleCount_; i++){

        osg::Vec3f spitze = osg::Vec3f(0,0,1);
        osg::Matrix spray_pos = owner_->getMatrix();
        osg::Vec3 duese = spray_pos.getTrans();
        spitze = spray_pos*spitze;

        float randAngle = ((float)rand())/(float)randMax;

        float offset = -10;                                                 //Abstand Duesenmitte zur emittierenden Seite
        float sprayAngle = sprayAngle_*Pi/180*randAngle*0.5;
        float massRand = ((float)rand())/(float)randMax;											  //random value between 0 and 1

        x[i] = duese.x()+spitze.x()*offset;
        y[i] = duese.y()+spitze.y()*offset;
        z[i] = duese.z()+spitze.z()*offset;

        r[i] = 0.000025+0.00005*massRand;
        m[i] = 4/3*pow(r[i],3)*Pi*1000;

        float v = getInitPressure()*pow(r[i],2)*Pi/m[i];                           //Beschleunigung des Partikels
        float d_angle = (float)rand()/(float)randMax*2*Pi;
        //if(i == 20)printf("%f\n", v);

        vx[i] = v*sin(sprayAngle)*cos(d_angle);
        vz[i] = v*sin(sprayAngle)*sin(d_angle);
        vy[i] = v*cos(sprayAngle);

        osg::Vec3f buffer = osg::Vec3f(vx[i],vy[i],vz[i]);

//        osg::Matrix spray_normal;
//        spray_normal.orthoNormal(spray_pos);

        osg::Quat a = spray_pos.getRotate();
        osg::Matrix spray_rot;
        spray_rot.setRotate(a);

        buffer = /*spray_normal**/spray_rot*buffer;

        vx[i] = buffer.x();
        vy[i] = buffer.y();
        vz[i] = buffer.z();

    }
    setCoSphere();
}
