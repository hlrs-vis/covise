#ifndef Raytracer_H
#define Raytracer_H

#include <embree3/rtcore.h>
#include <embree3/rtcore_device.h>
#include <embree3/rtcore_geometry.h>
#include <embree3/rtcore_ray.h>
#include <embree3/rtcore_common.h>
#include <osg/Array>
#include <list>
#include <vector>
class particle;
#define PROGRESS_SIZE 99999999


class Raytracer
{
private:

    struct Vertex { float x, y, z; };        
    struct Triangle { int v0, v1, v2; };      

    static Raytracer* _instance;
    Raytracer(){}
    Raytracer(const Raytracer&);
    ~Raytracer();

    RTCDevice gDevice = rtcNewDevice("start_threads=1,set_affinity=1,hugepages=1");
    RTCScene rScene_ = nullptr;
    std::list<RTCGeometry> geoList;
    std::list<unsigned int> geoIDList;
    bool comitted = false;
    int numRays = 0;

    RTCRayHit* x;
    RTCIntersectContext* d;
    osg::Vec3 vTemp;

public:
    static Raytracer* instance();
    void init();
    int addGeometry(RTCGeometry geo);
    void removeGeometry(int geomID);
    void removeAllGeometry();
    void finishAddGeometry();
    int getNumRays();
    void setNumRays(int newNumRays);
    int createCube(osg::Vec3 center, osg::Vec3 scale);
    int createFace(osg::Vec3Array::iterator coords, int type /*0 = triangle, 1 = quad*/);
    int createFace(osg::Vec3 v1, osg::Vec3 v2, osg::Vec3 v3, int type /*0 = triangle, 1 = quad*/);
    int createFaceSet(osg::Vec3Array* coords, int type = 0); //type = 0 for triangles, type = 1 for quads
    void checkAllHits(float time);

};



#endif // Raytracer_H
