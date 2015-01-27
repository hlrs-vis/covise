/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * surfel.hpp
 *
 *  Created on: Sep 4, 2012
 *      Author: asher
 */

#ifndef OSGPCL_SURFEL_HPP_
#define OSGPCL_SURFEL_HPP_

#include <surfel.h>
#include <impl/point_cloud.hpp>

template <typename PointT, typename NormalT>
osgpcl::SurfelFactory<PointT, NormalT>::SurfelFactory(float radius)
{

    const char *vertShader = {
        "#version 120\n"
        "void main(){\n"
        "    gl_Position    = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
        "    gl_FrontColor  = gl_Color;\n"
        "    gl_TexCoord[0] = gl_MultiTexCoord0;\n"
        "}\n"
    };
    const char *geomShader = {
        "#version 120\n"
        "#extension GL_EXT_geometry_shader4 : enable\n"
        "#define PI_6  3.1456/6\n"
        "#define PI_3  3.1456/3\n"
        "#define PI2 3.1456*2\n"
        "uniform float radius;\n"
        "void main( void )\n"
        "{\n"
        "for( int i = 0 ; i < 1 ; i++ ){\n"
        "vec4 center =  gl_PositionIn  [ i ]; \n"
        "gl_FrontColor  = gl_FrontColorIn[ i ]; \n"
        "gl_TexCoord[0] = gl_TexCoordIn  [ i ][ 0 ];\n"
        "gl_Position = center;\n"
        "EmitVertex();\n"
        "gl_Position = center+ gl_ModelViewProjectionMatrix*vec4(1.0  ,  0.0,0,0)*radius;\n"
        "EmitVertex();\n"
        "gl_Position = center+  gl_ModelViewProjectionMatrix*vec4(0.866025403784 , 0.5 ,0, 0)*radius;\n"
        "EmitVertex();\n"
        "for(float theta = PI_3; theta <= PI2; theta+= PI_6){\n"
        "    gl_Position = center;\n"
        "    EmitVertex();\n"
        "    gl_Position = center+  gl_ModelViewProjectionMatrix*vec4(cos(theta)  ,  sin(theta) , 0,0)*radius;\n"
        "    EmitVertex();\n"
        "}\n"
        "}\n"
        "}\n"
    };

    osg::ref_ptr<osg::Program> surfel_program = new osg::Program;
    surfel_program->setName("Surfel");
    surfel_program->addShader(new osg::Shader(osg::Shader::VERTEX, vertShader));
    surfel_program->addShader(new osg::Shader(osg::Shader::GEOMETRY, geomShader));
    surfel_program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3 + 24);
    surfel_program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
    surfel_program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    stateset_ = new osg::StateSet;
    stateset_->setAttribute(surfel_program);
    osg::Uniform *uradius(new osg::Uniform("radius", radius));
    stateset_->addUniform(uradius);
    stateset_->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
}

template <typename PointT, typename NormalT>
osg::Geometry *osgpcl::SurfelFactory<PointT, NormalT>::buildGeometry(bool unique)
{

    typename pcl::PointCloud<NormalT>::ConstPtr normals = this->getInputCloud<NormalT>();
    typename pcl::PointCloud<PointT>::ConstPtr xyz = this->getInputCloud<PointT>();

    if (xyz == NULL)
        return NULL;
    if (normals == NULL)
        return NULL;
    if (xyz->points.size() != normals->points.size())
        return NULL;
    pcl::IndicesConstPtr indices = indices_;
    {
        bool rebuild_indices = false;
        if (indices_ == NULL)
            rebuild_indices = true;
        else if (indices_->size() != xyz->points.size())
            rebuild_indices = true;
        if (rebuild_indices)
        {
            pcl::IndicesPtr idxs(new std::vector<int>);
            idxs->reserve(xyz->points.size());
            for (int i = 0; i < xyz->points.size(); i++)
                idxs->push_back(i);
            indices = idxs;
        }
    }

    osg::Vec3Array *pts = new osg::Vec3Array;
    osg::Vec3Array *npts = new osg::Vec3Array;
    pts->reserve(indices->size());
    npts->reserve(indices->size());
    for (int i = 0; i < indices->size(); i++)
    {
        const int &idx = (*indices)[i];
        const PointT &pt = xyz->points[idx];
        const NormalT &npt = normals->points[idx];
        pts->push_back(osg::Vec3(pt.x, pt.y, pt.z));
        npts->push_back(osg::Vec3(npt.normal_x, npt.normal_y, npt.normal_z));
    }
    osg::Geometry *geom = new osg::Geometry;
    geom->setVertexArray(pts);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_POINTS, 0, pts->size()));

    geom->setNormalArray(npts);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    geom->setStateSet(stateset_);
    return geom;
}

template <typename PointT, typename NormalT>
inline void osgpcl::SurfelFactory<PointT, NormalT>::setInputCloud(
    const pcl::PCLPointCloud2::ConstPtr &cloud)
{
    typename pcl::PointCloud<PointT>::Ptr xyz(new pcl::PointCloud<PointT>);
    pcl::fromPCLPointCloud2(*cloud, *xyz);
    PointCloudFactory::setInputCloud<PointT>(xyz);

    typename pcl::PointCloud<NormalT>::Ptr norms(new pcl::PointCloud<NormalT>);
    pcl::fromPCLPointCloud2(*cloud, *norms);
    PointCloudFactory::setInputCloud<NormalT>(norms);
}

template <typename PointT, typename NormalT>
void osgpcl::SurfelFactory<PointT, NormalT>::setRadius(float radius)
{
    stateset_->getUniform("radius")->set(radius);
}

// ****************************** SURFEL INTENSITY **************************************************

template <typename PointT, typename NormalT, typename IntensityT>
osgpcl::SurfelFactoryI<PointT, NormalT, IntensityT>::SurfelFactoryI(float radius)
{

    const char *vertShader = {
        "#version 120\n"
        "varying out vec3 normpt;\n"
        "void main(){\n"
        "    gl_Position    = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
        "    gl_FrontColor  = gl_Color;\n"
        "    gl_TexCoord[0] = gl_MultiTexCoord0;\n"
        "    normpt=gl_NormalMatrix * gl_Normal;\n"
        "}\n"
    };
    const char *geomShader = {
        "#version 120\n"
        "#extension GL_EXT_geometry_shader4 : enable\n"
        "#define PI_6  3.1456/6\n"
        "#define PI_3  3.1456/3\n"
        "#define PI2 3.1456*2\n"
        "uniform float radius;\n"
        "varying in vec3 normpt[1];\n"
        "void main( void )\n"
        "{\n"
        "vec3 normal = normpt[0];\n"
        "mat4 nrot; nrot[0]= vec4(1,0,0,0); nrot[1] =vec4(0,1,0,0); nrot[2]= vec4(0,0,1,0); nrot[3]= vec4(0,0,0,1);\n"
        "nrot[2][0]=normal[0];  nrot[2][1]=normal[1]; nrot[2][2]=normal[2];\n"
        "vec3 ax  ; if (abs(normal[0])>0.5) { ax =vec3(0,1,0) ;} else ax = vec3(1,0,0); \n"
        "ax = normalize( ax -normal*dot(ax,normal));  nrot[0][0]=ax[0]; nrot[0][1]=ax[1]; nrot[0][2]=ax[2];\n"
        "ax = normalize(cross(normal,ax));  nrot[1][0]=ax[0]; nrot[1][1]=ax[1]; nrot[1][2]=ax[2];\n"
        "for( int i = 0 ; i < 1 ; i++ ){\n"
        "vec4 center =  gl_PositionIn  [ i ]; \n"
        "gl_FrontColor  = gl_FrontColorIn[ i ]; \n"
        "gl_TexCoord[0] = gl_TexCoordIn  [ i ][ 0 ];\n"
        "gl_Position = center;\n"
        "EmitVertex();\n"
        "gl_Position = center+ nrot*vec4(1.0  ,  0.0,0,0)*radius;\n"
        "EmitVertex();\n"
        "gl_Position = center+  nrot*vec4(0.866025403784 , 0.5 ,0, 0)*radius;\n"
        "EmitVertex();\n"
        "for(float theta = PI_3; theta <= PI2; theta+= PI_6){\n"
        "    gl_Position = center;\n"
        "    EmitVertex();\n"
        "    gl_Position = center+  nrot*vec4(cos(theta)  ,  sin(theta) , 0,0)*radius;\n"
        "    EmitVertex();\n"
        "}\n"
        "}\n"
        "}\n"
    };

    osg::ref_ptr<osg::Program> surfel_program = new osg::Program;
    surfel_program->setName("Surfel");
    surfel_program->addShader(new osg::Shader(osg::Shader::VERTEX, vertShader));
    surfel_program->addShader(new osg::Shader(osg::Shader::GEOMETRY, geomShader));
    surfel_program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3 + 24);
    surfel_program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
    surfel_program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);

    stateset_ = new osg::StateSet;
    stateset_->setAttribute(surfel_program);
    osg::Uniform *uradius(new osg::Uniform("radius", radius));
    stateset_->addUniform(uradius);
    stateset_->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
}

template <typename PointT, typename NormalT, typename IntensityT>
osg::Geometry *osgpcl::SurfelFactoryI<PointT, NormalT, IntensityT>::buildGeometry(bool unique)
{

    typename pcl::PointCloud<NormalT>::ConstPtr normals = this->getInputCloud<NormalT>();
    typename pcl::PointCloud<PointT>::ConstPtr xyz = this->getInputCloud<PointT>();
    typename pcl::PointCloud<IntensityT>::ConstPtr iten = this->getInputCloud<IntensityT>();

    if (xyz == NULL)
        return NULL;
    if (normals == NULL)
        return NULL;
    if (iten == NULL)
        return NULL;

    if (xyz->points.size() != normals->points.size())
        return NULL;
    if (xyz->points.size() != iten->points.size())
        return NULL;

    pcl::IndicesConstPtr indices = indices_;
    {
        bool rebuild_indices = false;
        if (indices_ == NULL)
            rebuild_indices = true;
        else if (indices_->size() != xyz->points.size())
            rebuild_indices = true;
        if (rebuild_indices)
        {
            pcl::IndicesPtr idxs(new std::vector<int>);
            idxs->reserve(xyz->points.size());
            for (int i = 0; i < xyz->points.size(); i++)
                idxs->push_back(i);
            indices = idxs;
        }
    }

    osg::Vec3Array *pts = new osg::Vec3Array;
    osg::Vec3Array *npts = new osg::Vec3Array;
    osg::Vec3Array *cpts = new osg::Vec3Array;
    pts->reserve(indices->size());
    npts->reserve(indices->size());
    npts->reserve(indices->size());
    for (int i = 0; i < indices->size(); i++)
    {
        const int &idx = (*indices)[i];
        const PointT &pt = xyz->points[idx];
        const NormalT &npt = normals->points[idx];
        const IntensityT &ipt = iten->points[idx];
        pts->push_back(osg::Vec3(pt.x, pt.y, pt.z));
        npts->push_back(osg::Vec3(npt.normal_x, npt.normal_y, npt.normal_z));
        cpts->push_back(osg::Vec3(ipt.intensity, ipt.intensity, ipt.intensity));
    }
    osg::Geometry *geom = new osg::Geometry;
    geom->setVertexArray(pts);
    geom->addPrimitiveSet(new osg::DrawArrays(GL_POINTS, 0, pts->size()));

    geom->setNormalArray(npts);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    geom->setColorArray(cpts);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    geom->setStateSet(stateset_);
    return geom;
}

template <typename PointT, typename NormalT, typename IntensityT>
inline void osgpcl::SurfelFactoryI<PointT, NormalT, IntensityT>::setInputCloud(
    const pcl::PCLPointCloud2::ConstPtr &cloud)
{
    typename pcl::PointCloud<PointT>::Ptr xyz(new pcl::PointCloud<PointT>);
    pcl::fromPCLPointCloud2(*cloud, *xyz);
    PointCloudFactory::setInputCloud<PointT>(xyz);

    typename pcl::PointCloud<NormalT>::Ptr norms(new pcl::PointCloud<NormalT>);
    pcl::fromPCLPointCloud2(*cloud, *norms);
    PointCloudFactory::setInputCloud<NormalT>(norms);

    typename pcl::PointCloud<IntensityT>::Ptr iten(new pcl::PointCloud<IntensityT>);
    pcl::fromPCLPointCloud2(*cloud, *iten);
    PointCloudFactory::setInputCloud<IntensityT>(iten);
}

template <typename PointT, typename NormalT, typename IntensityT>
void osgpcl::SurfelFactoryI<PointT, NormalT, IntensityT>::setRadius(float radius)
{
    stateset_->getUniform("radius")->set(radius);
}
//***************************************************** Surfel FFI ******************//

template <typename PointT, typename NormalT, typename IntensityT>
osgpcl::SurfelFactoryFFI<PointT, NormalT, IntensityT>::SurfelFactoryFFI(float radius)
{
    stateset_ = new osg::StateSet;
    stateset_->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    radius_ = radius;

    int n = 8;
    circle_cache.resize(n + 1, 3);

    float theta = 2.0 * M_PI / (float)n;
    for (int i = 0; i <= n; i++)
    {
        circle_cache.row(i) << cos(theta * i) * radius, sin(theta * i) * radius, 0;
    }
}

template <typename PointT, typename NormalT, typename IntensityT>
osg::Geometry *osgpcl::SurfelFactoryFFI<PointT, NormalT, IntensityT>::buildGeometry(bool unique)
{

    typename pcl::PointCloud<NormalT>::ConstPtr normals = this->getInputCloud<NormalT>();
    typename pcl::PointCloud<PointT>::ConstPtr xyz = this->getInputCloud<PointT>();
    typename pcl::PointCloud<IntensityT>::ConstPtr iten = this->getInputCloud<IntensityT>();

    if (xyz == NULL)
        return NULL;
    if (normals == NULL)
        return NULL;
    if (iten == NULL)
        return NULL;

    if (xyz->points.size() != normals->points.size())
        return NULL;
    if (xyz->points.size() != iten->points.size())
        return NULL;

    pcl::IndicesConstPtr indices = indices_;
    {
        bool rebuild_indices = false;
        if (indices_ == NULL)
            rebuild_indices = true;
        else if (indices_->size() != xyz->points.size())
            rebuild_indices = true;
        if (rebuild_indices)
        {
            pcl::IndicesPtr idxs(new std::vector<int>);
            idxs->reserve(xyz->points.size());
            for (int i = 0; i < xyz->points.size(); i++)
                idxs->push_back(i);
            indices = idxs;
        }
    }

    osg::Vec3Array *pts = new osg::Vec3Array;
    osg::Vec3Array *npts = new osg::Vec3Array;
    osg::Vec3Array *cpts = new osg::Vec3Array;

    int fan_size = 1 + circle_cache.rows();
    pts->reserve(indices->size() * fan_size);
    npts->reserve(indices->size() * fan_size);
    cpts->reserve(indices->size());

    osg::Geometry *geom = new osg::Geometry;

    for (int i = 0, pstart = 0; i < indices->size(); i++)
    {
        const int &idx = (*indices)[i];
        const PointT &pt = xyz->points[idx];
        const NormalT &npt = normals->points[idx];
        const IntensityT &ipt = iten->points[idx];

        pts->push_back(osg::Vec3(pt.x, pt.y, pt.z));
        npts->push_back(osg::Vec3(npt.normal_x, npt.normal_y, npt.normal_z));
        cpts->push_back(osg::Vec3(ipt.intensity, ipt.intensity, ipt.intensity));

        pcl::Normal nt;

        Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();
        rot.row(2) << npt.getNormalVector3fMap().transpose();
        Eigen::Vector3f ax2(1, 0, 0);
        if (npt.getNormalVector3fMap().dot(ax2) > 0.1)
        {
            ax2 << 0, 1, 0;
            if (npt.getNormalVector3fMap().dot(ax2) > 0.1)
            {
                ax2 << 0, 0, 1;
            }
        }
        rot.row(1) << ax2.cross(npt.getNormalVector3fMap()).normalized().transpose();
        rot.row(0) = (ax2 - ax2.dot(npt.getNormalVector3fMap()) * npt.getNormalVector3fMap()).normalized();

        for (int j = 0; j < circle_cache.rows(); j++)
        {
            Eigen::Vector3f apt = rot * circle_cache.row(j).transpose() + pt.getVector3fMap();
            pts->push_back(osg::Vec3(apt[0], apt[1], apt[2]));
            npts->push_back(osg::Vec3(npt.normal_x, npt.normal_y, npt.normal_z));
        }
        geom->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLE_FAN, pstart, fan_size));
        pstart += fan_size;
    }
    geom->setVertexArray(pts);
    geom->setNormalArray(npts);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setColorArray(cpts);
    geom->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE_SET);
    geom->setStateSet(stateset_);
    return geom;
}

template <typename PointT, typename NormalT, typename IntensityT>
inline void osgpcl::SurfelFactoryFFI<PointT, NormalT, IntensityT>::setInputCloud(
    const pcl::PCLPointCloud2::ConstPtr &cloud)
{
    typename pcl::PointCloud<PointT>::Ptr xyz(new pcl::PointCloud<PointT>);
    pcl::fromPCLPointCloud2(*cloud, *xyz);
    PointCloudFactory::setInputCloud<PointT>(xyz);

    typename pcl::PointCloud<NormalT>::Ptr norms(new pcl::PointCloud<NormalT>);
    pcl::fromPCLPointCloud2(*cloud, *norms);
    PointCloudFactory::setInputCloud<NormalT>(norms);

    typename pcl::PointCloud<IntensityT>::Ptr iten(new pcl::PointCloud<IntensityT>);
    pcl::fromPCLPointCloud2(*cloud, *iten);
    PointCloudFactory::setInputCloud<IntensityT>(iten);
}

template <typename PointT, typename NormalT, typename IntensityT>
void osgpcl::SurfelFactoryFFI<PointT, NormalT, IntensityT>::setRadius(float radius)
{
    radius_ = radius;
}
// *********************************************************************************88

template <typename PointT, typename NormalT, typename RadiusT>
osgpcl::SurfelFactoryFF<PointT, NormalT, RadiusT>::SurfelFactoryFF()
{
    stateset_ = new osg::StateSet;
    stateset_->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    int n = 8;
    circle_cache.resize(n + 1, 3);

    float theta = 2.0 * M_PI / (float)n;
    for (int i = 0; i <= n; i++)
    {
        circle_cache.row(i) << cos(theta * i), sin(theta * i), 0;
    }
}

template <typename PointT, typename NormalT, typename RadiusT>
void osgpcl::SurfelFactoryFF<PointT, NormalT, RadiusT>::setInputCloud(
    const pcl::PCLPointCloud2::ConstPtr &cloud)
{
    typename pcl::PointCloud<PointT>::Ptr xyz(new pcl::PointCloud<PointT>);
    pcl::fromPCLPointCloud2(*cloud, *xyz);
    PointCloudFactory::setInputCloud<PointT>(xyz);

    if (!boost::is_same<PointT, NormalT>::value)
    {
        typename pcl::PointCloud<NormalT>::Ptr norms(new pcl::PointCloud<NormalT>);
        pcl::fromPCLPointCloud2(*cloud, *norms);
        PointCloudFactory::setInputCloud<NormalT>(norms);
    }
    if (!boost::is_same<PointT, RadiusT>::value)
    {
        typename pcl::PointCloud<RadiusT>::Ptr norms(new pcl::PointCloud<RadiusT>);
        pcl::fromPCLPointCloud2(*cloud, *norms);
        PointCloudFactory::setInputCloud<RadiusT>(norms);
    }
}

template <typename PointT, typename NormalT, typename RadiusT>
osgpcl::PointCloudGeometry *osgpcl::SurfelFactoryFF<PointT, NormalT, RadiusT>::buildGeometry(
    bool unique_state)
{

    typename pcl::PointCloud<NormalT>::ConstPtr normals = this->getInputCloud<NormalT>();
    typename pcl::PointCloud<PointT>::ConstPtr xyz = this->getInputCloud<PointT>();
    typename pcl::PointCloud<RadiusT>::ConstPtr rads = this->getInputCloud<RadiusT>();

    if (xyz == NULL)
        return NULL;
    if (normals == NULL)
        return NULL;
    if (rads == NULL)
        return NULL;

    if (xyz->points.size() != normals->points.size())
        return NULL;
    if (rads->points.size() != normals->points.size())
        return NULL;

    pcl::IndicesConstPtr indices = indices_;
    {
        bool rebuild_indices = false;
        if (indices_ == NULL)
            rebuild_indices = true;
        else if (indices_->size() != xyz->points.size())
            rebuild_indices = true;
        if (rebuild_indices)
        {
            pcl::IndicesPtr idxs(new std::vector<int>);
            idxs->reserve(xyz->points.size());
            for (int i = 0; i < xyz->points.size(); i++)
                idxs->push_back(i);
            indices = idxs;
        }
    }

    osg::Vec3Array *pts = new osg::Vec3Array;
    osg::Vec3Array *npts = new osg::Vec3Array;
    int fan_size = 1 + circle_cache.rows();
    pts->reserve(indices->size() * fan_size);
    npts->reserve(indices->size() * fan_size);

    osg::Geometry *geom = new osg::Geometry;

    for (int i = 0, pstart = 0; i < indices->size(); i++)
    {
        const int &idx = (*indices)[i];
        const PointT &pt = xyz->points[idx];
        const NormalT &npt = normals->points[idx];
        pts->push_back(osg::Vec3(pt.x, pt.y, pt.z));
        npts->push_back(osg::Vec3(npt.normal_x, npt.normal_y, npt.normal_z));

        pcl::Normal nt;

        Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();
        rot.row(2) << npt.getNormalVector3fMap().transpose();
        Eigen::Vector3f ax2(1, 0, 0);
        if (npt.getNormalVector3fMap().dot(ax2) > 0.1)
        {
            ax2 << 0, 1, 0;
            if (npt.getNormalVector3fMap().dot(ax2) > 0.1)
            {
                ax2 << 0, 0, 1;
            }
        }
        rot.row(1) << ax2.cross(npt.getNormalVector3fMap()).normalized().transpose();
        rot.row(0) = (ax2 - ax2.dot(npt.getNormalVector3fMap()) * npt.getNormalVector3fMap()).normalized();
        rot = rot * rads->points[idx].radius;

        for (int j = 0; j < circle_cache.rows(); j++)
        {
            Eigen::Vector3f apt = rot * circle_cache.row(j).transpose() + pt.getVector3fMap();
            pts->push_back(osg::Vec3(apt[0], apt[1], apt[2]));
            npts->push_back(osg::Vec3(npt.normal_x, npt.normal_y, npt.normal_z));
        }
        geom->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLE_FAN, pstart, fan_size));
        pstart += fan_size;
    }
    geom->setVertexArray(pts);
    geom->setNormalArray(npts);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    geom->setStateSet(stateset_);
    return geom;
}

#endif /* SURFEL_HPP_ */
