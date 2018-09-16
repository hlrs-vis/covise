#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <embree3/rtcore.h>
#include <embree3/rtcore_device.h>
#include <embree3/rtcore_geometry.h>
#include <embree3/rtcore_ray.h>
#include <embree3/rtcore_common.h>
#include <osg/Array>
#include <list>
#include <vector>
#include "types.h"
#include "gen.h"

struct Vertex   { float x,y,z/*,r*/;  };        //From tutorial
struct Triangle { int v0, v1, v2; };        //From tutorial

class raytracer
{
private:
    static raytracer* _instance;
    raytracer(){}
    raytracer(const raytracer&);
    ~raytracer()
    {
        removeAllGeometry();
        rtcReleaseScene(rScene_);
        rtcReleaseDevice(gDevice);
    }

    RTCDevice gDevice = rtcNewDevice("hughpages=1");
    RTCScene rScene_ = nullptr;
    std::list<RTCGeometry> geoList;
    std::list<unsigned int> geoIDList;
    bool comitted = false;

public:
    static raytracer* instance()
    {
        if(!_instance) _instance = new raytracer;
        return _instance;
    }

    void init()
    {
        rScene_ = rtcNewScene(gDevice);
    }

    int addGeometry(RTCGeometry geo)
    {
        if(comitted)comitted = false;
        geoList.push_back(geo);
        unsigned int geomID = rtcAttachGeometry(rScene_,geo);
        return geomID;
    }

    void removeGeometry(int geomID)
    {
        if(comitted)comitted = false;
        rtcDetachGeometry(rScene_, geomID);
    }

    void removeAllGeometry()
    {
        if(comitted)
            comitted = false;
        if(!geoIDList.empty())
        {
            for(auto i = geoIDList.begin(); i != geoIDList.end(); i++)
                rtcDetachGeometry(rScene_,(*i));
            geoList.clear();
        }

        finishAddGeometry();
    }

    void finishAddGeometry()
    {
        comitted = true;
        rtcCommitScene(rScene_);
    }

    int createCube(osg::Vec3 center, osg::Vec3 scale)
    {
        scale = osg::Vec3(scale.x()*0.5,scale.y()*0.5, scale.z()*0.5);
        /* create a triangulated cube with 12 triangles and 8 vertices */
        RTCGeometry mesh = rtcNewGeometry(gDevice, RTC_GEOMETRY_TYPE_TRIANGLE);

        /* set vertices */
        Vertex* vertices = (Vertex*) rtcSetNewGeometryBuffer(mesh,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,sizeof(Vertex),8);

        vertices[0].x = -1*scale.x()+center.x(); vertices[0].y = -1*scale.y()+center.y(); vertices[0].z = -1*scale.z()+center.z();
        vertices[1].x = -1*scale.x()+center.x(); vertices[1].y = -1*scale.y()+center.y(); vertices[1].z = +1*scale.z()+center.z();
        vertices[2].x = -1*scale.x()+center.x(); vertices[2].y = +1*scale.y()+center.y(); vertices[2].z = -1*scale.z()+center.z();
        vertices[3].x = -1*scale.x()+center.x(); vertices[3].y = +1*scale.y()+center.y(); vertices[3].z = +1*scale.z()+center.z();
        vertices[4].x = +1*scale.x()+center.x(); vertices[4].y = -1*scale.y()+center.y(); vertices[4].z = -1*scale.z()+center.z();
        vertices[5].x = +1*scale.x()+center.x(); vertices[5].y = -1*scale.y()+center.y(); vertices[5].z = +1*scale.z()+center.z();
        vertices[6].x = +1*scale.x()+center.x(); vertices[6].y = +1*scale.y()+center.y(); vertices[6].z = -1*scale.z()+center.z();
        vertices[7].x = +1*scale.x()+center.x(); vertices[7].y = +1*scale.y()+center.y(); vertices[7].z = +1*scale.z()+center.z();

        Triangle* triangles = (Triangle*) rtcSetNewGeometryBuffer(mesh,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,sizeof(Triangle),12);
        int tri = 0;
        // left side
        triangles[tri].v0 = 0; triangles[tri].v1 = 1; triangles[tri].v2 = 2; tri++;
        triangles[tri].v0 = 1; triangles[tri].v1 = 3; triangles[tri].v2 = 2; tri++;

        // right side
        triangles[tri].v0 = 4; triangles[tri].v1 = 6; triangles[tri].v2 = 5; tri++;
        triangles[tri].v0 = 5; triangles[tri].v1 = 6; triangles[tri].v2 = 7; tri++;

        // bottom side
        triangles[tri].v0 = 0; triangles[tri].v1 = 4; triangles[tri].v2 = 1; tri++;
        triangles[tri].v0 = 1; triangles[tri].v1 = 4; triangles[tri].v2 = 5; tri++;

        // top side
        triangles[tri].v0 = 2; triangles[tri].v1 = 3; triangles[tri].v2 = 6; tri++;
        triangles[tri].v0 = 3; triangles[tri].v1 = 7; triangles[tri].v2 = 6; tri++;

        // front side
        triangles[tri].v0 = 0; triangles[tri].v1 = 2; triangles[tri].v2 = 4; tri++;
        triangles[tri].v0 = 2; triangles[tri].v1 = 6; triangles[tri].v2 = 4; tri++;

        // back side
        triangles[tri].v0 = 1; triangles[tri].v1 = 5; triangles[tri].v2 = 3; tri++;
        triangles[tri].v0 = 3; triangles[tri].v1 = 5; triangles[tri].v2 = 7; tri++;


        rtcSetGeometryVertexAttributeCount(mesh,1);
        //rtcSetSharedGeometryBuffer(mesh,RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE,0,RTC_FORMAT_FLOAT3,vertices,0,sizeof(Vertex),8);

        rtcCommitGeometry(mesh);
        if(comitted)comitted = false;
        geoList.push_back(mesh);
        unsigned int geomID = rtcAttachGeometry(rScene_,mesh);
        rtcReleaseGeometry(mesh);
        return geomID;
    }

    int createFace(osg::Vec3Array::iterator coords, int type /*0 = triangle, 1 = quad*/)
    {
        //printf("Creating face in embree\n");
        int numOfVertices = 0;
        RTCGeometryType geoType;
        if(type > 2)
            return -1;
        else if(type == 0)
        {
            numOfVertices = 3;
            geoType = RTC_GEOMETRY_TYPE_TRIANGLE;
        }
        else
        {
            numOfVertices = 4;
            geoType = RTC_GEOMETRY_TYPE_QUAD;
        }

        RTCGeometry mesh = rtcNewGeometry(gDevice, geoType);

        Vertex* vertices = (Vertex*) rtcSetNewGeometryBuffer(mesh,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,sizeof(Vertex),numOfVertices);

        for(int i = 0; i< numOfVertices; i++)
        {
            osg::Vec3 buf = *coords;
            //printf("%f %f %f\n", buf.x(), buf.y(), buf.z());
            vertices[i].x = buf.x();
            vertices[i].y = buf.z();
            vertices[i].z = buf.y();
            coords++;
        }
        int numOfFaces = 0;
        if(type == 0)
            numOfFaces = 1;
        else
            numOfFaces = 2;
        Triangle* triangles = (Triangle*) rtcSetNewGeometryBuffer(mesh,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,sizeof(Triangle),numOfFaces);

        triangles[0] = {0,1,2};
        if(type == 1) triangles[1] = {0,2,3};

        rtcSetGeometryVertexAttributeCount(mesh,1);

        rtcCommitGeometry(mesh);
        if(comitted)comitted = false;
        geoList.push_back(mesh);
        unsigned int geomID = rtcAttachGeometry(rScene_,mesh);
        rtcReleaseGeometry(mesh);
        return geomID;


    }

    int createFace(osg::Vec3 v1, osg::Vec3 v2, osg::Vec3 v3, int type /*0 = triangle, 1 = quad*/)
    {
        //printf("Creating face in embree\n");
        int numOfVertices = 0;
        if(type > 2)
            return -1;
        else if(type == 0)
            numOfVertices = 3;
        else
            numOfVertices = 4;

        RTCGeometry mesh = rtcNewGeometry(gDevice, RTC_GEOMETRY_TYPE_TRIANGLE);

        Vertex* vertices = (Vertex*) rtcSetNewGeometryBuffer(mesh,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,sizeof(Vertex),numOfVertices);


            //printf("%f %f %f\n", buf.x(), buf.y(), buf.z());
            vertices[0].x = v1.x();
            vertices[0].y = v1.z();
            vertices[0].z = v1.y();
            vertices[1].x = v2.x();
            vertices[1].y = v2.z();
            vertices[1].z = v2.y();
            vertices[2].x = v3.x();
            vertices[2].y = v3.z();
            vertices[2].z = v3.y();


        int numOfFaces = 0;
        if(type == 0)
            numOfFaces = 1;
        else
            numOfFaces = 2;
        Triangle* triangles = (Triangle*) rtcSetNewGeometryBuffer(mesh,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,sizeof(Triangle),numOfFaces);

        triangles[0] = {0,1,2};
        if(type == 1) triangles[1] = {0,2,3};

        rtcSetGeometryVertexAttributeCount(mesh,1);

        rtcCommitGeometry(mesh);
        if(comitted)comitted = false;
        geoList.push_back(mesh);
        unsigned int geomID = rtcAttachGeometry(rScene_,mesh);
        rtcReleaseGeometry(mesh);
        return geomID;


    }

    int createFaceSet(osg::Vec3Array* coords, int type) //type = 0 for triangles, type = 1 for quads
    {
        //printf("Creating face in embree\n");
        int numOfVertices = 0;
        RTCGeometryType geoType;
        if(type > 2)
            return -1;
        else if(type == 0)
        {
            numOfVertices = 3;
            geoType = RTC_GEOMETRY_TYPE_TRIANGLE;
        }
        else
        {
            numOfVertices = 4;
            geoType = RTC_GEOMETRY_TYPE_QUAD;
        }

        RTCGeometry mesh = rtcNewGeometry(gDevice, geoType);

        Vertex* vertices = (Vertex*) rtcSetNewGeometryBuffer(mesh,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,sizeof(Vertex),coords->size());

        osg::Vec3Array::iterator itr = coords->begin();

        for(int i = 0; i< coords->size(); i++)
        {
            osg::Vec3 buf = *itr;
            vertices[i].x = buf.x();
            vertices[i].y = buf.z();
            vertices[i].z = buf.y();
            itr++;
        }
        int numOfFaces = 0;
        if(type == 0)
            numOfFaces = coords->size()/3;
        else
            numOfFaces = coords->size()/4;
        Triangle* triangles = (Triangle*) rtcSetNewGeometryBuffer(mesh,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,sizeof(Triangle),numOfFaces);

        if(type == 0)
            for(int fItr = 0; fItr < numOfFaces; fItr++)
            {
                triangles[fItr] = {fItr*numOfVertices, fItr*numOfVertices+1, fItr*numOfVertices+2};
            }
        if(type == 1)
            for(int fItr = 0; fItr < numOfFaces; fItr+=2)
            {
                triangles[fItr] = {fItr*numOfVertices, fItr*numOfVertices+1, fItr*numOfVertices+2};
                triangles[fItr+1] = {fItr*numOfVertices, fItr*numOfVertices+2, fItr*numOfVertices+3};
            }

        std::cout << "size " << coords->size() << std::endl;

        rtcSetGeometryVertexAttributeCount(mesh,1);

        rtcCommitGeometry(mesh);
        if(comitted)
            comitted = false;
        geoList.push_back(mesh);
        unsigned int geomID = rtcAttachGeometry(rScene_,mesh);
        geoIDList.push_back(geomID);
        rtcReleaseGeometry(mesh);
        return geomID;


    }

    float checkForHit(particle p, float time)
    {
        RTCRayHit x;
        p.velocity *= time;
        x.ray.org_x = p.pos.x();
        x.ray.org_y = p.pos.z();
        x.ray.org_z = p.pos.y();
        x.ray.dir_x = p.velocity.x();
        x.ray.dir_y = p.velocity.z();
        x.ray.dir_z = p.velocity.y();
        x.ray.tfar = 1;
        x.ray.flags = 0;
        x.ray.tnear = 0;
        x.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        x.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        x.hit.instID[1] = RTC_INVALID_GEOMETRY_ID;

        RTCIntersectContext d;
        d.flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
        rtcInitIntersectContext(&d);
        rtcIntersect1(rScene_,&d,&x);

        if(x.hit.geomID != -1)
        {
            return x.ray.tfar;
        }
        else
        {
            return -1;
        }
    }

};



#endif // RAYTRACER_H
