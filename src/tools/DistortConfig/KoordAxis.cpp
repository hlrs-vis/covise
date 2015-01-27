/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KoordAxis.h"

KoordAxis::KoordAxis(void)
{
}

KoordAxis::~KoordAxis(void)
{
}

osgText::Text *KoordAxis::createAxisLabel(const std::string &iLabel, const osg::Vec3 &iPosition)
{
    osg::ref_ptr<osgText::Text> txtLabel = new osgText::Text;
    txtLabel->setFont("arial.ttf");
    txtLabel->setText(iLabel);
    txtLabel->setPosition(iPosition);
    txtLabel->setCharacterSize(17);
    txtLabel->setAutoRotateToScreen(true);
    txtLabel->setColor(osg::Vec4(204, 204, 0, 1));
    txtLabel->setCharacterSizeMode(osgText::Text::SCREEN_COORDS);
    return txtLabel.release();
}

osg::Geometry *KoordAxis::createArrow(const osg::Matrixd &iTransform, const osg::Vec4 &iColor, double iHeight)
{
    osg::ref_ptr<osg::Geometry> geoArrow = new osg::Geometry;

    double pyramidBaseZ = iHeight / 3.0 * 2.0;
    double outerBaseRadius = iHeight / 9.0;
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array(7);
    (*vertices)[0].set(iTransform.preMult(osg::Vec3d(outerBaseRadius, 0.0, pyramidBaseZ)));
    (*vertices)[1].set(iTransform.preMult(osg::Vec3d(0.0, outerBaseRadius, pyramidBaseZ)));
    (*vertices)[2].set(iTransform.preMult(osg::Vec3d(-outerBaseRadius, 0.0, pyramidBaseZ)));
    (*vertices)[3].set(iTransform.preMult(osg::Vec3d(0.0, -outerBaseRadius, pyramidBaseZ)));
    (*vertices)[4].set(iTransform.preMult(osg::Vec3d(0.0, 0.0, iHeight)));
    (*vertices)[5].set(iTransform.preMult(osg::Vec3d(0.0, 0.0, iHeight)));
    (*vertices)[6].set(iTransform.preMult(osg::Vec3d(0.0, 0.0, 0.0)));

    osg::ref_ptr<osg::UByteArray> indices = new osg::UByteArray(20);
    (*indices)[0] = 0;
    (*indices)[1] = 1;
    (*indices)[2] = 4;
    (*indices)[3] = 1;
    (*indices)[4] = 2;
    (*indices)[5] = 4;
    (*indices)[6] = 2;
    (*indices)[7] = 3;
    (*indices)[8] = 4;
    (*indices)[9] = 3;
    (*indices)[10] = 0;
    (*indices)[11] = 4;
    (*indices)[12] = 1;
    (*indices)[13] = 0;
    (*indices)[14] = 3;
    (*indices)[15] = 2;
    (*indices)[16] = 1;
    (*indices)[17] = 3;
    (*indices)[18] = 5;
    (*indices)[19] = 6;

    geoArrow->setVertexArray(vertices.get());
    geoArrow->setVertexIndices(indices.get());

    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
    colors->push_back(iColor);
    geoArrow->setColorArray(colors.get());
    geoArrow->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::ref_ptr<osg::DrawArrays> drawArray1 = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, indices->size() - 2);
    geoArrow->addPrimitiveSet(drawArray1.get());

    osg::ref_ptr<osg::DrawArrays> drawArray2 = new osg::DrawArrays(osg::PrimitiveSet::LINES, indices->size() - 2, 2);
    geoArrow->addPrimitiveSet(drawArray2.get());

    return geoArrow.release();
}

osg::Geometry *KoordAxis::createXAxis(double iHeight)
{
    osg::Matrixd transform = osg::Matrix::rotate(osg::inDegrees(90.0f), 0.0f, 1.0f, 0.0f);
    osg::Vec4 color(0.5f, 0.125f, 0.125f, 1.0f);
    osg::ref_ptr<osg::Geometry> geoXAxis = createArrow(transform, color, iHeight);
    return geoXAxis.release();
}

osg::Geometry *KoordAxis::createYAxis(double iHeight)
{
    osg::Matrixd transform = osg::Matrix::rotate(osg::inDegrees(-90.0f), 1.0f, 0.0f, 0.0f);
    osg::Vec4 color(0.125f, 0.5f, 0.125f, 1.0f);
    osg::ref_ptr<osg::Geometry> geoYAxis = createArrow(transform, color, iHeight);
    return geoYAxis.release();
}

osg::Geometry *KoordAxis::createZAxis(double iHeight)
{
    osg::Matrixd transform = osg::Matrix::identity();
    osg::Vec4 color(0.125f, 0.125f, 0.5f, 1.0f);
    osg::ref_ptr<osg::Geometry> geoZAxis = createArrow(transform, color, iHeight);
    return geoZAxis.release();
}

osg::Geode *KoordAxis::createAxesGeometry(double length)
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode;

    osg::ref_ptr<osg::LineWidth> lineWidth = new osg::LineWidth(2);
    geode->getOrCreateStateSet()->setAttributeAndModes(lineWidth.get(), osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    geode->addDrawable(createXAxis(length));
    geode->addDrawable(createYAxis(length));
    geode->addDrawable(createZAxis(length));
    geode->addDrawable(createAxisLabel("X", osg::Vec3(length * 1.05, 0, 0)));
    geode->addDrawable(createAxisLabel("Y", osg::Vec3(0, length * 1.05, 0)));
    geode->addDrawable(createAxisLabel("Z", osg::Vec3(0, 0, length * 1.05)));

    return geode.release();
}
