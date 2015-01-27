/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * point_cloud.hpp
 *
 *  Created on: Jul 27, 2012
 *      Author: Adam Stambler
 */

#ifndef POINT_CLOUD_HPP_
#define POINT_CLOUD_HPP_

#include <point_cloud.h>
#include <pcl/console/print.h>
#include <osg/Point>
#include <osg/LineWidth>
#include <pcl/point_traits.h>
#include <pcl/common/concatenate.h>
#include <pcl/conversions.h>
#include <pcl/common/common.h>

#include <boost/type_traits.hpp>

#include <stdlib.h>
#include <time.h>

template <typename PointT>
inline typename pcl::PointCloud<PointT>::ConstPtr osgpcl::PointCloudFactory::getInputCloud() const
{

    std::string key; // Get the fields list
    {
        pcl::PointCloud<PointT> cloud;
        key = pcl::getFieldsList(cloud);
    }

    std::map<std::string, boost::any>::const_iterator iter = input_clouds_.find(key);

    if (iter == input_clouds_.end())
    {
        pcl::console::print_error("PointCloudFactory trying to retrieve input %s that does not exist\n", key.c_str());
        return typename pcl::PointCloud<PointT>::ConstPtr();
    }
    try
    {
        return boost::any_cast<typename pcl::PointCloud<PointT>::ConstPtr>(iter->second);
    }
    catch (boost::bad_any_cast &e)
    {
        pcl::console::print_error("PointCloudFactory Exception: %s\n", e.what());
        return typename pcl::PointCloud<PointT>::ConstPtr();
    }
}

template <typename PointT>
inline void osgpcl::PointCloudFactory::addXYZToVertexBuffer(osg::Geometry &geom,
                                                            const pcl::PointCloud<pcl::PointXYZ> &cloud) const
{
    osg::Vec3Array *pts = new osg::Vec3Array;
    pts->reserve(cloud.points.size());
    for (int i = 0; i < cloud.points.size(); i++)
    {
        const PointT &pt = cloud.points[i];
        pts->push_back(osg::Vec3(pt.x, pt.y, pt.z));
    }
    geom.setVertexArray(pts);
    geom.addPrimitiveSet(new osg::DrawArrays(GL_POINTS, 0, pts->size()));
}

// *****************************  PointCloudColorFactory ******************

template <typename PointT>
inline osgpcl::PointCloudColoredFactory<PointT>::PointCloudColoredFactory()
{
    const char *vertSource = {
        "#version 120\n"
        "void main(void)\n"
        "{\n"
        "    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
        "}\n"
    };
    const char *fragSource = {
        "#version 120\n"
        "uniform vec4 color;\n"
        "void main(void)\n"
        "{\n"
        "    gl_FragColor = color;\n"
        "}\n"
    };

    osg::Program *pgm = new osg::Program;
    pgm->setName("UniformColor");

    pgm->addShader(new osg::Shader(osg::Shader::VERTEX, vertSource));
    pgm->addShader(new osg::Shader(osg::Shader::FRAGMENT, fragSource));
    stateset_ = new osg::StateSet;
    stateset_->setAttribute(pgm);

    osg::Point *p = new osg::Point;
    p->setSize(4);

    stateset_->setAttribute(p);

    osg::Vec4 color;
    color[0] = color[1] = color[2] = color[3] = 1;

    osg::Uniform *ucolor(new osg::Uniform("color", color));
    stateset_->addUniform(ucolor);

    stateset_->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
}

template <typename PointT>
osgpcl::PointCloudGeometry *
osgpcl::PointCloudColoredFactory<PointT>::buildGeometry(bool unique_stateset)
{
    typename pcl::PointCloud<PointT>::ConstPtr cloud = getInputCloud<PointT>();
    if (cloud == NULL)
        return NULL;

    PointCloudGeometry *geom = new PointCloudGeometry;
    this->addXYZToVertexBuffer<PointT>(*geom, *cloud);

    osg::ref_ptr<osg::StateSet> ss;
    if (unique_stateset)
    {
        ss = new osg::StateSet(*stateset_);
    }
    else
    {
        ss = stateset_;
    }
    geom->setStateSet(stateset_);

    return geom;
}

template <typename PointT>
void
osgpcl::PointCloudColoredFactory<PointT>::setInputCloud(
    const pcl::PCLPointCloud2::ConstPtr &cloud)
{
    typename pcl::PointCloud<PointT>::Ptr xyz(new pcl::PointCloud<PointT>);
    pcl::fromPCLPointCloud2(*cloud, *xyz);
    PointCloudFactory::setInputCloud<PointT>(xyz);
}

template <typename PointT>
void
osgpcl::PointCloudColoredFactory<PointT>::setColor(float r, float g, float b,
                                                   float alpha)
{
    osg::Vec4 color;
    color[0] = r;
    color[1] = g;
    color[2] = b;
    color[3] = alpha;
    stateset_->getUniform("color")->set(color);
}

// *************************************** PointCloudRGBFactory *************************
template <typename PointTXYZ, typename RGBT>
osgpcl::PointCloudGeometry *
osgpcl::PointCloudRGBFactory<PointTXYZ, RGBT>::buildGeometry(
    bool unique_stateset)
{
    osgpcl::PointCloudGeometry *geom(new PointCloudGeometry);

    typename pcl::PointCloud<PointTXYZ>::ConstPtr xyz = getInputCloud<PointTXYZ>();
    typename pcl::PointCloud<RGBT>::ConstPtr rgb = getInputCloud<RGBT>();
    if ((rgb == NULL) || (xyz == NULL))
    {
        return NULL;
    }

    if (rgb->points.size() != xyz->points.size())
    {
        pcl::console::print_error("[PointCloudRGBFactory]  XYZ and Label Clouds have different # of points.\n");
        return NULL;
    }

    //TODO Make the color table a texture and then just
    // reference the texture inside the shader

    osg::Vec4Array *colors = new osg::Vec4Array;
    colors->reserve(rgb->points.size());
    int psize = rgb->points.size();

    srand(time(NULL));
    for (int i = 0; i < psize; i++)
    {
        osg::Vec4f c;
        c[0] = (float)rgb->points[i].r / 255.0f;
        c[1] = (float)rgb->points[i].g / 255.0f;
        c[2] = (float)rgb->points[i].b / 255.0f;
        c[3] = 1;
        colors->push_back(c);
    }

    this->addXYZToVertexBuffer<PointTXYZ>(*geom, *xyz);

    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    if (unique_stateset)
    {
        geom->setStateSet(new osg::StateSet(*stateset_));
    }
    else
    {
        geom->setStateSet(stateset_);
    }
    return geom;

    return geom;
}

template <typename PointTXYZ, typename RGBT>
inline void osgpcl::PointCloudRGBFactory<PointTXYZ, RGBT>::setInputCloud(
    const pcl::PCLPointCloud2::ConstPtr &cloud)
{
    typename pcl::PointCloud<PointTXYZ>::Ptr xyz(new pcl::PointCloud<PointTXYZ>);
    pcl::fromPCLPointCloud2(*cloud, *xyz);
    PointCloudFactory::setInputCloud<PointTXYZ>(xyz);

    if (!boost::is_same<PointTXYZ, RGBT>::value)
    {
        typename pcl::PointCloud<RGBT>::Ptr rgb(new pcl::PointCloud<RGBT>);
        pcl::fromPCLPointCloud2(*cloud, *rgb);
        PointCloudFactory::setInputCloud<RGBT>(rgb);
    }
}

// ******************************* PointCloudCRange *************************

template <typename PointTXYZ, typename PointTF>
inline osgpcl::PointCloudCRangeFactory<PointTXYZ, PointTF>::PointCloudCRangeFactory(std::string field)
    : max_range_(-1)
    , min_range_(-1)
    , field_name_(field)
{
    color_table_.push_back(osg::Vec4(1, 1, 1, 1));
    color_table_.push_back(osg::Vec4(1, 0, 0, 1));
    stateset_ = new osg::StateSet;
    setPointSize(4);
    stateset_->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
}

template <typename PointTXYZ, typename PointTF>
void
osgpcl::PointCloudCRangeFactory<PointTXYZ, PointTF>::setField(
    std::string field)
{
    field_name_ = field;
}

template <typename PointTXYZ, typename PointTF>
inline void osgpcl::PointCloudCRangeFactory<PointTXYZ, PointTF>::setRange(
    double min, double max)
{
    min_range_ = min;
    max_range_ = max;
}

template <typename PointTXYZ, typename PointTF>
inline void osgpcl::PointCloudCRangeFactory<PointTXYZ, PointTF>::setColorTable(
    const std::vector<osg::Vec4> &table)
{
    color_table_ = table;
}

template <typename PointTXYZ, typename PointTF>
osgpcl::PointCloudGeometry *
osgpcl::PointCloudCRangeFactory<PointTXYZ, PointTF>::buildGeometry(
    bool unique_stateset)
{

    typename pcl::PointCloud<PointTXYZ>::ConstPtr xyz = getInputCloud<PointTXYZ>();
    typename pcl::PointCloud<PointTF>::ConstPtr fcloud = getInputCloud<PointTF>();
    if ((fcloud == NULL) || (xyz == NULL))
    {
        return NULL;
    }
    double minr, maxr;

    std::vector<pcl::PCLPointField> flist;
    pcl::getFields<PointTF>(*fcloud, flist);

    int idx = -1;

    if (field_name_.empty())
    {
        idx = 0;
    }
    else
    {
        for (int i = 0; i < flist.size(); i++)
        {
            if (flist[i].name == field_name_)
            {
                idx = i;
                break;
            }
        }
    }

    if (idx < 0)
    {
        pcl::console::print_debug("[PointCloudCRangefactory] Pointfield ( %s )does not exist\n", field_name_.c_str());
        return NULL;
    }
    int offset = flist[idx].offset;

    if (fabs(min_range_ - max_range_) < 0.001)
    {
        minr = std::numeric_limits<double>::infinity();
        maxr = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < fcloud->points.size(); i++)
        {
            double val = *((float *)(((uint8_t *)&fcloud->points[i]) + offset));
            if (val < minr)
                minr = val;
            if (val > maxr)
                maxr = val;
        }
    }
    else
    {
        minr = min_range_;
        maxr = max_range_;
    }
    double scale = (color_table_.size() - 1) / (maxr - minr);
    int maxidx = color_table_.size() - 1;

    osg::Vec4Array *colors = new osg::Vec4Array;
    colors->resize(fcloud->points.size());
    int psize = fcloud->points.size();
    for (int i = 0; i < psize; i++)
    {
        double val = *((float *)(((uint8_t *)&fcloud->points[i]) + offset));
        double idx = (val - minr) * scale;
        if (idx < 0)
            idx = 0;
        if (idx > maxidx)
            idx = maxidx;
        double wl = idx - std::floor(idx);
        double wu = 1 - wl;
        const osg::Vec4f &lpt = color_table_[std::floor(idx)];
        const osg::Vec4f &upt = color_table_[std::ceil(idx)];
        for (int j = 0; j < 4; j++)
            (*colors)[i][j] = lpt[j] * wl + upt[j] * wu;
    }

    PointCloudGeometry *geom = new PointCloudGeometry;

    this->addXYZToVertexBuffer<PointTXYZ>(*geom, *xyz);

    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    if (unique_stateset)
    {
        geom->setStateSet(new osg::StateSet(*stateset_));
    }
    else
    {
        geom->setStateSet(stateset_);
    }

    return geom;
}

template <typename PointTXYZ, typename PointTF>
void
osgpcl::PointCloudCRangeFactory<PointTXYZ, PointTF>::setPointSize(
    int size)
{
    osg::Point *p = new osg::Point();
    p->setSize(4);
    stateset_->setAttribute(p);
}

template <typename PointTXYZ, typename PointTF>
inline void osgpcl::PointCloudCRangeFactory<PointTXYZ, PointTF>::setInputCloud(
    const pcl::PCLPointCloud2::ConstPtr &cloud)
{
    typename pcl::PointCloud<PointTXYZ>::Ptr xyz(new pcl::PointCloud<PointTXYZ>);
    pcl::fromPCLPointCloud2(*cloud, *xyz);
    PointCloudFactory::setInputCloud<PointTXYZ>(xyz);

    if (!boost::is_same<PointTXYZ, PointTF>::value)
    {
        typename pcl::PointCloud<PointTF>::Ptr fcloud(new pcl::PointCloud<PointTF>);
        pcl::fromPCLPointCloud2(*cloud, *fcloud);
        PointCloudFactory::setInputCloud<PointTF>(fcloud);
    }
}

//**************************************** Intensity Point Cloud *******************

template <typename PointTXYZ, typename IntensityT>
osgpcl::PointCloudGeometry *
osgpcl::PointCloudIFactory<PointTXYZ, IntensityT>::buildGeometry(bool unique_stateset)
{
    typename pcl::PointCloud<PointTXYZ>::ConstPtr xyz = getInputCloud<PointTXYZ>();
    typename pcl::PointCloud<IntensityT>::ConstPtr icloud = getInputCloud<IntensityT>();
    if ((icloud == NULL) || (xyz == NULL))
    {
        return NULL;
    }

    if (icloud->points.size() != xyz->points.size())
    {
        pcl::console::print_error("[PointCloudIntensityFactory]  XYZ and Label Clouds have different # of points.\n");
        return NULL;
    }

    //TODO just make this a single grayscale value and make a custom shader program
    osg::Vec4Array *colors = new osg::Vec4Array;
    colors->reserve(icloud->points.size());
    int psize = icloud->points.size();

    for (int i = 0; i < psize; i++)
    {
        osg::Vec4f color;
        color[0] = color[1] = color[2] = icloud->points[i].intensity * 0.8 + 0.2;
        color[3] = 1;
        colors->push_back(color);
    }

    PointCloudGeometry *geom = new PointCloudGeometry;

    this->addXYZToVertexBuffer<PointTXYZ>(*geom, *xyz);

    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setStateSet(stateset_);
    return geom;
}

template <typename PointTXYZ, typename IntensityT>
void osgpcl::PointCloudIFactory<PointTXYZ, IntensityT>::setInputCloud(
    const pcl::PCLPointCloud2::ConstPtr &cloud)
{
    typename pcl::PointCloud<PointTXYZ>::Ptr xyz(new pcl::PointCloud<PointTXYZ>);
    pcl::fromPCLPointCloud2(*cloud, *xyz);
    PointCloudFactory::setInputCloud<PointTXYZ>(xyz);

    if (!boost::is_same<PointTXYZ, IntensityT>::value)
    {
        typename pcl::PointCloud<IntensityT>::Ptr icloud(new pcl::PointCloud<IntensityT>);
        pcl::fromPCLPointCloud2(*cloud, *icloud);
        PointCloudFactory::setInputCloud<IntensityT>(icloud);
    }
}

//******************************* Label Point Cloud **********************************/
template <typename PointTXYZ, typename LabelT>
inline osgpcl::PointCloudLabelFactory<PointTXYZ, LabelT>::PointCloudLabelFactory()
{
    //set up a basic color map for consistency on the typical labels used
    color_map_[0] = osg::Vec4f(0.2, 0.2, 0.2, 1);
    color_map_[1] = osg::Vec4f(1, 0, 0, 1);
    color_map_[2] = osg::Vec4f(0, 1, 0, 1);
    color_map_[3] = osg::Vec4f(0, 0, 1, 1);
    color_map_[4] = osg::Vec4f(1, 0, 1, 1);
    color_map_[5] = osg::Vec4f(1, 1, 0, 1);
    color_map_[6] = osg::Vec4f(0.4, 0.1, 0.1, 1);
    color_map_[7] = osg::Vec4f(0.4, 0.4, 0.1, 1);
    color_map_[8] = osg::Vec4f(0.4, 0.1, 0.4, 1);
    color_map_[9] = osg::Vec4f(0.1, 0.8, 0.1, 1);
    color_map_[10] = osg::Vec4f(0.8, 0.1, 0.1, 1);
    color_map_[11] = osg::Vec4f(0.8, 0.8, 0.1, 1);
    color_map_[12] = osg::Vec4f(0.8, 0.1, 0.8, 1);
    color_map_[13] = osg::Vec4f(0.1, 0.8, 0.1, 1);
    color_map_[14] = osg::Vec4f(0.6, 0.3, 0.3, 1);
    color_map_[15] = osg::Vec4f(0.6, 0.6, 0.3, 1);
    color_map_[16] = osg::Vec4f(0.6, 0.3, 0.6, 1);
    color_map_[17] = osg::Vec4f(0.3, 0.6, 0.3, 1);

    random_coloring_ = true;

    stateset_ = new osg::StateSet;
    stateset_->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    osg::Point *p = new osg::Point();
    p->setSize(4);
    stateset_->setAttribute(p);
}

template <typename PointTXYZ, typename LabelT>
inline osgpcl::PointCloudGeometry *osgpcl::PointCloudLabelFactory<PointTXYZ, LabelT>::buildGeometry(
    bool unique_stateset)
{

    typename pcl::PointCloud<PointTXYZ>::ConstPtr xyz = getInputCloud<PointTXYZ>();
    typename pcl::PointCloud<LabelT>::ConstPtr lcloud = getInputCloud<LabelT>();
    if ((lcloud == NULL) || (xyz == NULL))
    {
        return NULL;
    }

    if (lcloud->points.size() != xyz->points.size())
    {
        pcl::console::print_error("[PointCloudLabelFactory]  XYZ and Label Clouds have different # of points.\n");
        return NULL;
    }

    //TODO Make the color table a texture and then just
    // reference the texture inside the shader

    osg::Vec4Array *colors = new osg::Vec4Array;
    colors->reserve(lcloud->points.size());
    int psize = lcloud->points.size();

    ColorMap &cmap = color_map_;
    srand(time(NULL));
    for (int i = 0; i < psize; i++)
    {
        ColorMap::iterator iter = cmap.find(lcloud->points[i].label);
        if (iter == cmap.end())
        {
            osg::Vec4f color;
            if (random_coloring_)
            {
                for (int i = 0; i < 3; i++)
                    color[i] = ((float)(rand() % 900)) / 900.0f + 0.1;
            }
            else
            {
                color = osg::Vec4f(0, 0, 0, 1);
            }
            color[3] = 1;
            cmap[lcloud->points[i].label] = color;
            colors->push_back(color);
        }
        else
        {
            colors->push_back(iter->second);
        }
    }

    PointCloudGeometry *geom = new PointCloudGeometry;
    this->addXYZToVertexBuffer<PointTXYZ>(*geom, *xyz);

    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    if (unique_stateset)
    {
        geom->setStateSet(new osg::StateSet(*stateset_));
    }
    else
    {
        geom->setStateSet(stateset_);
    }
    return geom;
}

template <typename PointTXYZ, typename LabelT>
inline void osgpcl::PointCloudLabelFactory<PointTXYZ, LabelT>::setInputCloud(
    const pcl::PCLPointCloud2::ConstPtr &cloud)
{
    typename pcl::PointCloud<PointTXYZ>::Ptr xyz(new pcl::PointCloud<PointTXYZ>);
    pcl::fromPCLPointCloud2(*cloud, *xyz);
    PointCloudFactory::setInputCloud<PointTXYZ>(xyz);

    if (!boost::is_same<PointTXYZ, LabelT>::value)
    {
        typename pcl::PointCloud<LabelT>::Ptr icloud(new pcl::PointCloud<LabelT>);
        pcl::fromPCLPointCloud2(*cloud, *icloud);
        PointCloudFactory::setInputCloud<LabelT>(icloud);
    }
}

template <typename PointTXYZ, typename LabelT>
inline void osgpcl::PointCloudLabelFactory<PointTXYZ, LabelT>::setColorMap(
    const ColorMap &color_map)
{
    color_map_ = color_map;
}

template <typename PointTXYZ, typename LabelT>
inline void osgpcl::PointCloudLabelFactory<PointTXYZ, LabelT>::enableRandomColoring(
    bool enable)
{
    random_coloring_ = enable;
}

// ********************************* Normal Factory ********************************

template <typename PointTXYZ, typename NormalT>
inline osgpcl::PointCloudNormalFactory<PointTXYZ, NormalT>::PointCloudNormalFactory()
{
    stateset_ = new osg::StateSet;

    /*osg::Point* p = new osg::Point;
    p->setSize(6);
    stateset_->setAttribute(p);
    */

    osg::LineWidth *lw = new osg::LineWidth;
    lw->setWidth(4);
    stateset_->setAttribute(lw);

    stateset_->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    nlength = 0.5;
    color_[0] = 1;
    color_[1] = 1;
    color_[2] = 0;
    color_[3] = 1;
}

template <typename PointTXYZ, typename NormalT>
inline osgpcl::PointCloudGeometry *osgpcl::PointCloudNormalFactory<PointTXYZ, NormalT>::buildGeometry(
    bool unique_stateset)
{

    typename pcl::PointCloud<PointTXYZ>::ConstPtr xyz = getInputCloud<PointTXYZ>();
    typename pcl::PointCloud<NormalT>::ConstPtr normals = getInputCloud<NormalT>();
    if ((normals == NULL) || (xyz == NULL))
    {
        return NULL;
    }

    if (normals->points.size() != xyz->points.size())
    {
        pcl::console::print_error("[PointCloudNormalFactory]  XYZ and Normal Clouds have different # of points.\n");
        return NULL;
    }

    osg::Geometry *geom = new osg::Geometry;
    osg::Vec3Array *pts = new osg::Vec3Array;
    pts->reserve(xyz->points.size());
    for (int i = 0; i < xyz->points.size(); i++)
    {
        const PointTXYZ &pt = xyz->points[i];
        pts->push_back(osg::Vec3(pt.x, pt.y, pt.z));
        Eigen::Vector3f npt = pt.getVector3fMap() + normals->points[i].getNormalVector3fMap() * nlength;
        pts->push_back(osg::Vec3(npt[0], npt[1], npt[2]));
    }
    geom->setVertexArray(pts);
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, xyz->points.size()));
    osg::Vec4Array *colors = new osg::Vec4Array;
    colors->push_back(color_);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    if (unique_stateset)
    {
        geom->setStateSet(new osg::StateSet(*stateset_));
    }
    else
    {
        geom->setStateSet(stateset_);
    }
    return geom;
}

template <typename PointTXYZ, typename NormalT>
inline void osgpcl::PointCloudNormalFactory<PointTXYZ, NormalT>::setInputCloud(
    const pcl::PCLPointCloud2::ConstPtr &cloud)
{
    typename pcl::PointCloud<PointTXYZ>::Ptr xyz(new pcl::PointCloud<PointTXYZ>);
    pcl::fromPCLPointCloud2(*cloud, *xyz);
    PointCloudFactory::setInputCloud<PointTXYZ>(xyz);

    if (!boost::is_same<PointTXYZ, NormalT>::value)
    {
        typename pcl::PointCloud<NormalT>::Ptr icloud(new pcl::PointCloud<NormalT>);
        pcl::fromPCLPointCloud2(*cloud, *icloud);
        PointCloudFactory::setInputCloud<NormalT>(icloud);
    }
}

template <typename PointTXYZ, typename NormalT>
inline void osgpcl::PointCloudNormalFactory<PointTXYZ, NormalT>::setColor(
    osg::Vec4f color)
{
    color_ = color;
}

#endif /* POINT_CLOUD_HPP_ */
