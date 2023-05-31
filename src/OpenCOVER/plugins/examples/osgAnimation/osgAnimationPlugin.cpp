/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: osgAnimation Plugin (skin Animation exaple)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "osgAnimationPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

#include <iostream>
#include <osgDB/ReadFile>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/TrackballManipulator>
#include <osgGA/FlightManipulator>
#include <osgGA/DriveManipulator>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/StateSetManipulator>
#include <osgGA/AnimationPathManipulator>
#include <osgGA/TerrainManipulator>
#include <osg/Drawable>
#include <osg/MatrixTransform>

#include <osgAnimation/BasicAnimationManager>
#include <osgAnimation/RigGeometry>
#include <osgAnimation/RigTransformHardware>
#include <osgAnimation/AnimationManagerBase>
#include <osgAnimation/BoneMapVisitor>
#include <osgAnimation/Bone>
#include <osgAnimation/Skeleton>
#include <osgAnimation/UpdateMatrixTransform>
#include <osgAnimation/UpdateBone>
#include <osgAnimation/StackedTransform>
#include <osgAnimation/StackedTranslateElement>
#include <osgAnimation/StackedRotateAxisElement>

#include <sstream>

osg::Geode *createAxis()
{
    osg::Geode *geode(new osg::Geode());
    osg::Geometry *geometry(new osg::Geometry());

    osg::Vec3Array *vertices(new osg::Vec3Array());
    vertices->push_back(osg::Vec3(0.0, 0.0, 0.0));
    vertices->push_back(osg::Vec3(1.0, 0.0, 0.0));
    vertices->push_back(osg::Vec3(0.0, 0.0, 0.0));
    vertices->push_back(osg::Vec3(0.0, 1.0, 0.0));
    vertices->push_back(osg::Vec3(0.0, 0.0, 0.0));
    vertices->push_back(osg::Vec3(0.0, 0.0, 1.0));
    geometry->setVertexArray(vertices);

    osg::Vec4Array *colors(new osg::Vec4Array());
    colors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    colors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    colors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f));
    colors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f));
    colors->push_back(osg::Vec4(0.0f, 0.0f, 1.0f, 1.0f));
    colors->push_back(osg::Vec4(0.0f, 0.0f, 1.0f, 1.0f));
    geometry->setColorArray(colors);

    geometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 6));

    geode->addDrawable(geometry);
    return geode;
}

osgAnimation::RigGeometry *createTesselatedBox(int nsplit, float size)
{
    osgAnimation::RigGeometry *riggeometry = new osgAnimation::RigGeometry;

    osg::Geometry *geometry = new osg::Geometry;
    osg::ref_ptr<osg::Vec3Array> vertices(new osg::Vec3Array());
    osg::ref_ptr<osg::Vec3Array> colors(new osg::Vec3Array());
    geometry->setVertexArray(vertices.get());
    geometry->setColorArray(colors.get());
    geometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    float step = size / nsplit;
    float s = 0.5 / 4.0;
    for (int i = 0; i < nsplit; i++)
    {
        float x = -1 + i * step;
        std::cout << x << std::endl;
        vertices->push_back(osg::Vec3(x, s, s));
        vertices->push_back(osg::Vec3(x, -s, s));
        vertices->push_back(osg::Vec3(x, -s, -s));
        vertices->push_back(osg::Vec3(x, s, -s));
        osg::Vec3 c(0, 0, 0);
        c[i % 3] = 1;
        colors->push_back(c);
        colors->push_back(c);
        colors->push_back(c);
        colors->push_back(c);
    }

    osg::ref_ptr<osg::UIntArray> array = new osg::UIntArray;
    for (int i = 0; i < nsplit - 1; i++)
    {
        int base = i * 4;
        array->push_back(base);
        array->push_back(base + 1);
        array->push_back(base + 4);
        array->push_back(base + 1);
        array->push_back(base + 5);
        array->push_back(base + 4);

        array->push_back(base + 3);
        array->push_back(base);
        array->push_back(base + 4);
        array->push_back(base + 7);
        array->push_back(base + 3);
        array->push_back(base + 4);

        array->push_back(base + 5);
        array->push_back(base + 1);
        array->push_back(base + 2);
        array->push_back(base + 2);
        array->push_back(base + 6);
        array->push_back(base + 5);

        array->push_back(base + 2);
        array->push_back(base + 3);
        array->push_back(base + 7);
        array->push_back(base + 6);
        array->push_back(base + 2);
        array->push_back(base + 7);
    }

    geometry->addPrimitiveSet(new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES, array->size(), &array->front()));
    geometry->setUseDisplayList(false);
    riggeometry->setSourceGeometry(geometry);
    return riggeometry;
}

void initVertexMap(osgAnimation::Bone *b0,
                   osgAnimation::Bone *b1,
                   osgAnimation::Bone *b2,
                   osgAnimation::RigGeometry *geom,
                   osg::Vec3Array *array)
{
    osgAnimation::VertexInfluenceSet vertexesInfluences;
    osgAnimation::VertexInfluenceMap *vim = new osgAnimation::VertexInfluenceMap;

    (*vim)[b0->getName()].setName(b0->getName());
    (*vim)[b1->getName()].setName(b1->getName());
    (*vim)[b2->getName()].setName(b2->getName());

    for (int i = 0; i < (int)array->size(); i++)
    {
        float val = (*array)[i][0];
        std::cout << val << std::endl;
        if (val >= -1 && val <= 0)
            (*vim)[b0->getName()].push_back(osgAnimation::VertexIndexWeight(i, 1));
        else if (val > 0 && val <= 1)
            (*vim)[b1->getName()].push_back(osgAnimation::VertexIndexWeight(i, 1));
        else if (val > 1)
            (*vim)[b2->getName()].push_back(osgAnimation::VertexIndexWeight(i, 1));
    }

    geom->setInfluenceMap(vim);
}

static unsigned int getRandomValueinRange(unsigned int v)
{
    return static_cast<unsigned int>((rand() * 1.0 * v) / (RAND_MAX - 1));
}

osg::ref_ptr<osg::Program> program;
// show how to override the default RigTransformHardware for customized usage
struct MyRigTransformHardware : public osgAnimation::RigTransformHardware
{

    void operator()(osgAnimation::RigGeometry &geom)
    {
        if (_needInit)
            if (!init(geom))
                return;
        computeMatrixPaletteUniform(geom.getMatrixFromSkeletonToGeometry(), geom.getInvMatrixFromSkeletonToGeometry());
    }

    bool init(osgAnimation::RigGeometry &geom)
    {
        osg::Vec3Array *pos = dynamic_cast<osg::Vec3Array *>(geom.getVertexArray());
        if (!pos)
        {
            osg::notify(osg::WARN) << "RigTransformHardware no vertex array in the geometry " << geom.getName() << std::endl;
            return false;
        }

        if (!geom.getSkeleton())
        {
            osg::notify(osg::WARN) << "RigTransformHardware no skeleting set in geometry " << geom.getName() << std::endl;
            return false;
        }

        osgAnimation::BoneMapVisitor mapVisitor;
        geom.getSkeleton()->accept(mapVisitor);
        osgAnimation::BoneMap bm = mapVisitor.getBoneMap();

        if (!createPalette(pos->size(), bm, geom.getVertexInfluenceSet().getVertexToBoneList()))
            return false;

        int attribIndex = 11;
        int nbAttribs = getNumVertexAttrib();

        // use a global program for all avatar
        if (!program.valid())
        {
            program = new osg::Program;
            program->setName("HardwareSkinning");
            if (!_shader.valid())
                _shader = osg::Shader::readShaderFile(osg::Shader::VERTEX, "shaders/skinning.vert");

            if (!_shader.valid())
            {
                osg::notify(osg::WARN) << "RigTransformHardware can't load VertexShader" << std::endl;
                return false;
            }

            // replace max matrix by the value from uniform
            {
                std::string str = _shader->getShaderSource();
                std::string toreplace = std::string("MAX_MATRIX");
                std::size_t start = str.find(toreplace);
                std::stringstream ss;
                ss << getMatrixPaletteUniform()->getNumElements();
                str.replace(start, toreplace.size(), ss.str());
                _shader->setShaderSource(str);
                osg::notify(osg::INFO) << "Shader " << str << std::endl;
            }

            program->addShader(_shader.get());

            for (int i = 0; i < nbAttribs; i++)
            {
                std::stringstream ss;
                ss << "boneWeight" << i;
                program->addBindAttribLocation(ss.str(), attribIndex + i);

                osg::notify(osg::INFO) << "set vertex attrib " << ss.str() << std::endl;
            }
        }
        for (int i = 0; i < nbAttribs; i++)
        {
            std::stringstream ss;
            ss << "boneWeight" << i;
            geom.setVertexAttribData(attribIndex + i, osg::Geometry::ArrayData(getVertexAttrib(i), osg::Geometry::BIND_PER_VERTEX));
        }

        osg::ref_ptr<osg::StateSet> ss = new osg::StateSet;
        ss->addUniform(getMatrixPaletteUniform());
        ss->addUniform(new osg::Uniform("nbBonesPerVertex", getNumBonesPerVertex()));
        ss->setAttributeAndModes(program.get());
        geom.setStateSet(ss.get());
        _needInit = false;
        return true;
    }
};

struct SetupRigGeometry : public osg::NodeVisitor
{
    bool _hardware;
    SetupRigGeometry(bool hardware = true)
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
        , _hardware(hardware)
    {
    }

    void apply(osg::Geode &geode)
    {
        for (unsigned int i = 0; i < geode.getNumDrawables(); i++)
            apply(*geode.getDrawable(i));
    }
    void apply(osg::Drawable &geom)
    {
        if (_hardware)
        {
            osgAnimation::RigGeometry *rig = dynamic_cast<osgAnimation::RigGeometry *>(&geom);
            if (rig)
                rig->setRigTransformImplementation(new MyRigTransformHardware);
        }

#if 0
        if (geom.getName() != std::string("BoundingBox")) // we disable compute of bounding box for all geometry except our bounding box
            geom.setComputeBoundingBoxCallback(new osg::Drawable::ComputeBoundingBoxCallback);
//            geom.setInitialBound(new osg::Drawable::ComputeBoundingBoxCallback);
#endif
    }
};

osg::Group *createCharacterInstance(osg::Group *character, bool hardware)
{
    osg::ref_ptr<osg::Group> c;
    if (hardware)
        c = osg::clone(character, osg::CopyOp::DEEP_COPY_ALL & ~osg::CopyOp::DEEP_COPY_PRIMITIVES & ~osg::CopyOp::DEEP_COPY_ARRAYS);
    else
        c = osg::clone(character, osg::CopyOp::DEEP_COPY_ALL);

    osgAnimation::AnimationManagerBase *animationManager = dynamic_cast<osgAnimation::AnimationManagerBase *>(c->getUpdateCallback());

    osgAnimation::BasicAnimationManager *anim = dynamic_cast<osgAnimation::BasicAnimationManager *>(animationManager);
    const osgAnimation::AnimationList &list = animationManager->getAnimationList();
    int v = getRandomValueinRange(list.size());
    if (list[v]->getName() == std::string("MatIpo_ipo"))
    {
        anim->playAnimation(list[v].get());
        v = (v + 1) % list.size();
    }

    anim->playAnimation(list[v].get());

    SetupRigGeometry switcher(hardware);
    c->accept(switcher);

    return c.release();
}

osgAnimationPlugin::osgAnimationPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "osgAnimationPlugin::osgAnimationPlugin\n");

    std::cerr << "This example works better with nathan.osg" << std::endl;
    bool hardware = true;
    int maxChar = 10;

    osg::ref_ptr<osg::Group> root = dynamic_cast<osg::Group *>(osgDB::readNodeFile("nathan.osg"));
    if (!root)
    {
        std::cout << "osgAnimationPlugin: No data loaded" << std::endl;
        return;
    }

    osgAnimation::AnimationManagerBase *animationManager = dynamic_cast<osgAnimation::AnimationManagerBase *>(root->getUpdateCallback());
    if (!animationManager)
    {
        osg::notify(osg::FATAL) << "no AnimationManagerBase found, updateCallback need to animate elements" << std::endl;
        return;
    }

    double xChar = maxChar;
    double yChar = xChar * 9.0 / 16;
    for (double i = 0.0; i < xChar; i++)
    {
        for (double j = 0.0; j < yChar; j++)
        {

            osg::ref_ptr<osg::Group> c = createCharacterInstance(root.get(), hardware);
            osg::MatrixTransform *tr = new osg::MatrixTransform;
            tr->setMatrix(osg::Matrix::translate(2.0 * (i - xChar * .5),
                                                 0.0,
                                                 2.0 * (j - yChar * .5)));
            tr->addChild(c.get());
            cover->getObjectsRoot()->addChild(tr);
        }
    }
    std::cout << "created " << xChar *yChar << " instance" << std::endl;

    osg::ref_ptr<osgAnimation::Skeleton> skelroot = new osgAnimation::Skeleton;
    skelroot->setDefaultUpdateCallback();
    osg::ref_ptr<osgAnimation::Bone> rootBone = new osgAnimation::Bone;
    rootBone->setInvBindMatrixInSkeletonSpace(osg::Matrix::inverse(osg::Matrix::translate(-1, 0, 0)));
    rootBone->setName("root");
    osgAnimation::UpdateBone *pRootUpdate = new osgAnimation::UpdateBone("root");
    pRootUpdate->getStackedTransforms().push_back(new osgAnimation::StackedTranslateElement("translate", osg::Vec3(-1, 0, 0)));
    rootBone->setUpdateCallback(pRootUpdate);

    osg::ref_ptr<osgAnimation::Bone> right0 = new osgAnimation::Bone;
    right0->setInvBindMatrixInSkeletonSpace(osg::Matrix::inverse(osg::Matrix::translate(0, 0, 0)));
    right0->setName("right0");
    osgAnimation::UpdateBone *pRight0Update = new osgAnimation::UpdateBone("right0");
    pRight0Update->getStackedTransforms().push_back(new osgAnimation::StackedTranslateElement("translate", osg::Vec3(1, 0, 0)));
    pRight0Update->getStackedTransforms().push_back(new osgAnimation::StackedRotateAxisElement("rotate", osg::Vec3(0, 0, 1), 0));
    right0->setUpdateCallback(pRight0Update);

    osg::ref_ptr<osgAnimation::Bone> right1 = new osgAnimation::Bone;
    right1->setInvBindMatrixInSkeletonSpace(osg::Matrix::inverse(osg::Matrix::translate(1, 0, 0)));
    right1->setName("right1");
    osgAnimation::UpdateBone *pRight1Update = new osgAnimation::UpdateBone("right1");
    pRight1Update->getStackedTransforms().push_back(new osgAnimation::StackedTranslateElement("translate", osg::Vec3(1, 0, 0)));
    pRight1Update->getStackedTransforms().push_back(new osgAnimation::StackedRotateAxisElement("rotate", osg::Vec3(0, 0, 1), 0));
    right1->setUpdateCallback(pRight1Update);

    rootBone->addChild(right0.get());
    right0->addChild(right1.get());
    skelroot->addChild(rootBone.get());

    osg::ref_ptr<osgAnimation::BasicAnimationManager> manager = new osgAnimation::BasicAnimationManager;

    osgAnimation::Animation *anim = new osgAnimation::Animation;
    {
        osgAnimation::FloatKeyframeContainer *keys0 = new osgAnimation::FloatKeyframeContainer;
        keys0->push_back(osgAnimation::FloatKeyframe(0, 0));
        keys0->push_back(osgAnimation::FloatKeyframe(3, osg::PI_2));
        keys0->push_back(osgAnimation::FloatKeyframe(6, osg::PI_2));
        osgAnimation::FloatLinearSampler *sampler = new osgAnimation::FloatLinearSampler;
        sampler->setKeyframeContainer(keys0);
        osgAnimation::FloatLinearChannel *channel = new osgAnimation::FloatLinearChannel(sampler);
        channel->setName("rotate");
        channel->setTargetName("right0");
        anim->addChannel(channel);
    }

    {
        osgAnimation::FloatKeyframeContainer *keys1 = new osgAnimation::FloatKeyframeContainer;
        keys1->push_back(osgAnimation::FloatKeyframe(0, 0));
        keys1->push_back(osgAnimation::FloatKeyframe(3, 0));
        keys1->push_back(osgAnimation::FloatKeyframe(6, osg::PI_2));
        osgAnimation::FloatLinearSampler *sampler = new osgAnimation::FloatLinearSampler;
        sampler->setKeyframeContainer(keys1);
        osgAnimation::FloatLinearChannel *channel = new osgAnimation::FloatLinearChannel(sampler);
        channel->setName("rotate");
        channel->setTargetName("right1");
        anim->addChannel(channel);
    }
    manager->registerAnimation(anim);
    manager->buildTargetReference();

    // let's start !
    manager->playAnimation(anim);

    // we will use local data from the skeleton
    osg::MatrixTransform *rootTransform = new osg::MatrixTransform;
    rootTransform->setMatrix(osg::Matrix::rotate(osg::PI_2, osg::Vec3(1, 0, 0)));
    right0->addChild(createAxis());
    right0->setDataVariance(osg::Object::DYNAMIC);
    right1->addChild(createAxis());
    right1->setDataVariance(osg::Object::DYNAMIC);
    osg::MatrixTransform *trueroot = new osg::MatrixTransform;
    trueroot->setMatrix(osg::Matrix(rootBone->getMatrixInBoneSpace().ptr()));
    trueroot->addChild(createAxis());
    trueroot->addChild(skelroot.get());
    trueroot->setDataVariance(osg::Object::DYNAMIC);
    rootTransform->addChild(trueroot);

    rootTransform->setUpdateCallback(manager.get());
    cover->getObjectsRoot()->addChild(rootTransform);

    osgAnimation::RigGeometry *geom = createTesselatedBox(4, 4.0);
    osg::Geode *geode = new osg::Geode;
    geode->addDrawable(geom);
    skelroot->addChild(geode);
    osg::ref_ptr<osg::Vec3Array> src = dynamic_cast<osg::Vec3Array *>(geom->getSourceGeometry()->getVertexArray());
    geom->getOrCreateStateSet()->setMode(GL_LIGHTING, false);
    geom->setDataVariance(osg::Object::DYNAMIC);

    initVertexMap(rootBone.get(), right0.get(), right1.get(), geom, src.get());
}

// this is called if the plugin is removed at runtime
osgAnimationPlugin::~osgAnimationPlugin()
{
    fprintf(stderr, "osgAnimationPlugin::~osgAnimationPlugin\n");
}

void
osgAnimationPlugin::preFrame()
{
}

COVERPLUGIN(osgAnimationPlugin)
