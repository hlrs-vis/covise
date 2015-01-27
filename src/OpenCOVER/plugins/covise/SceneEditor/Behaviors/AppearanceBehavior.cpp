/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AppearanceBehavior.h"
#include "../SceneObject.h"

#include "../Events/SetAppearanceColorEvent.h"

#include <PluginUtil/ProgramCache.h>
#include <cover/VRSceneGraph.h>

#include <osg/StateSet>
#include <osgDB/ReadFile>

AppearanceBehavior::AppearanceBehavior()
{
    _type = BehaviorTypes::APPEARANCE_BEHAVIOR;
}

AppearanceBehavior::~AppearanceBehavior()
{
    for (std::vector<Scope>::iterator it = _scopeList.begin(); it != _scopeList.end(); it++)
    {
        if (it->appearance.shader != NULL)
            delete (it->appearance.shader);
    }
    ProgramCache::instance()->gc();
}

int AppearanceBehavior::attach(SceneObject *so)
{
    // connects this behavior to its scene object
    Behavior::attach(so);

    std::vector<Scope>::iterator it;
    for (it = _scopeList.begin(); it != _scopeList.end(); it++)
    {
        // shader for whole object
        if (it->regexp.isEmpty())
        {
            setAppearance(so->getGeometryNode(it->geoNameSpace), it->appearance, false);
        }
        else
        {
            QRegExp regexp(it->regexp);
            setAppearance(regexp, so->getGeometryNode(it->geoNameSpace), it->appearance, false);
        }
    }

    return 1;
}

int AppearanceBehavior::detach()
{
    std::vector<Scope>::iterator it;
    for (it = _scopeList.begin(); it != _scopeList.end(); it++)
    {
        // shader for whole object
        if (it->regexp.isEmpty())
        {
            setAppearance(_sceneObject->getGeometryNode(it->geoNameSpace), it->appearance, true);
        }
        else
        {
            QRegExp regexp(it->regexp);
            setAppearance(regexp, _sceneObject->getGeometryNode(it->geoNameSpace), it->appearance, true);
        }
    }

    ProgramCache::instance()->gc();
    Behavior::detach();

    return 1;
}

EventErrors::Type AppearanceBehavior::receiveEvent(Event *e)
{
    if (e->getType() == EventTypes::SET_APPEARANCE_COLOR_EVENT)
    {
        SetAppearanceColorEvent *sace = dynamic_cast<SetAppearanceColorEvent *>(e);
        std::string scopeName = sace->getScope();
        osg::Vec4 color = sace->getColor();

        Appearance appearance;
        std::vector<Scope>::iterator it;
        for (it = _scopeList.begin(); it != _scopeList.end(); it++)
        {
            if (it->name == scopeName)
            {
                appearance = it->appearance;
                break;
            }
        }

        appearance.color = color;
        appearance.material = createMaterial(color);

        if (scopeName == "")
        {
            setAppearance(_sceneObject->getGeometryNode(), appearance, false);
        }
        else
        {
            std::vector<Scope>::iterator it;
            for (it = _scopeList.begin(); it != _scopeList.end(); it++)
            {
                if (it->name == scopeName)
                {
                    QRegExp regexp(it->regexp);
                    setAppearance(regexp, _sceneObject->getGeometryNode(), appearance, false);
                }
            }
        }
        return EventErrors::SUCCESS;
    }

    return EventErrors::UNHANDLED;
}

void AppearanceBehavior::setAppearance(QRegExp regexp, osg::Node *node, Appearance appearance, bool remove)
{
    if (!node)
    {
        return;
    }
    if (regexp.exactMatch(QString(node->getName().c_str())))
    {
        setAppearance(node, appearance, remove);
    }
    else
    {
        osg::Group *g = node->asGroup();
        if (g != NULL)
        {
            for (int i = 0; i < g->getNumChildren(); i++)
            {
                setAppearance(regexp, g->getChild(i), appearance, remove);
            }
        }
    }
}

void AppearanceBehavior::setAppearance(osg::Node *n, Appearance appearance, bool remove)
{
    if (!n)
    {
        return;
    }
    osg::StateSet *sset = n->getOrCreateStateSet();
    if (remove)
    {
        if (appearance.shader)
        {
            sset->removeAttribute(osg::StateAttribute::PROGRAM);
            ProgramCache::instance()->gc();
        }
        if (appearance.material)
        {
            sset->removeAttribute(appearance.material);
        }
    }
    else
    {
        if (appearance.shader)
        {
            MyShader *s = appearance.shader;

            osg::ref_ptr<osg::Program> program = ProgramCache::instance()->getProgram(s->getVertex(), s->getFragment());
            sset->setAttributeAndModes(program, osg::StateAttribute::ON);

            // add uniform variable for material color to stateSet
            osg::Uniform *var = new osg::Uniform("MATERIAL_COLOR", appearance.color);
            sset->addUniform(var);
            // add uniform variables to stateSet
            std::map<std::string, int> intUni = s->getIntUniforms();
            std::map<std::string, int>::iterator itInt;
            for (itInt = intUni.begin(); itInt != intUni.end(); itInt++)
            {
                osg::Uniform *var = new osg::Uniform(itInt->first.c_str(), itInt->second);
                sset->addUniform(var);
            }
            std::map<std::string, bool> boolUni = s->getBoolUniforms();
            std::map<std::string, bool>::iterator itBool;
            for (itBool = boolUni.begin(); itBool != boolUni.end(); itBool++)
            {
                osg::Uniform *var = new osg::Uniform(itBool->first.c_str(), itBool->second);
                sset->addUniform(var);
            }
            std::map<std::string, float> floatUni = s->getFloatUniforms();
            std::map<std::string, float>::iterator itFloat;
            for (itFloat = floatUni.begin(); itFloat != floatUni.end(); itFloat++)
            {
                osg::Uniform *var = new osg::Uniform(itFloat->first.c_str(), itFloat->second);
                sset->addUniform(var);
            }
            std::map<std::string, osg::Vec2> vec2Uni = s->getVec2Uniforms();
            std::map<std::string, osg::Vec2>::iterator itVec2;
            for (itVec2 = vec2Uni.begin(); itVec2 != vec2Uni.end(); itVec2++)
            {
                osg::Uniform *var = new osg::Uniform(itVec2->first.c_str(), itVec2->second);
                sset->addUniform(var);
            }
            std::map<std::string, osg::Vec3> vec3Uni = s->getVec3Uniforms();
            std::map<std::string, osg::Vec3>::iterator itVec3;
            for (itVec3 = vec3Uni.begin(); itVec3 != vec3Uni.end(); itVec3++)
            {
                osg::Uniform *var = new osg::Uniform(itVec3->first.c_str(), itVec3->second);
                sset->addUniform(var);
            }
            std::map<std::string, osg::Vec4> vec4Uni = s->getVec4Uniforms();
            std::map<std::string, osg::Vec4>::iterator itVec4;
            for (itVec4 = vec4Uni.begin(); itVec4 != vec4Uni.end(); itVec4++)
            {
                osg::Uniform *var = new osg::Uniform(itVec4->first.c_str(), itVec4->second);
                sset->addUniform(var);
            }
            // now processing textures, starting with one reserved location for already existing textures
            int texture = 1;
            sset->addUniform(new osg::Uniform("EXISTING_TEXTURE", 0));
            // shader textures
            std::map<std::string, osg::ref_ptr<osg::Texture1D> > tex1DUni = s->getTexture1DUniforms();
            std::map<std::string, osg::ref_ptr<osg::Texture1D> >::iterator itTex1D;
            for (itTex1D = tex1DUni.begin(); itTex1D != tex1DUni.end(); itTex1D++)
            {
                sset->setTextureAttributeAndModes(texture, (itTex1D->second).get(), osg::StateAttribute::ON);
                osg::Uniform *var = new osg::Uniform(itTex1D->first.c_str(), texture);
                sset->addUniform(var);
                texture++;
            }
            std::map<std::string, osg::ref_ptr<osg::Texture2D> > tex2DUni = s->getTexture2DUniforms();
            std::map<std::string, osg::ref_ptr<osg::Texture2D> >::iterator itTex2D;
            for (itTex2D = tex2DUni.begin(); itTex2D != tex2DUni.end(); itTex2D++)
            {
                sset->setTextureAttributeAndModes(texture, itTex2D->second.get(), osg::StateAttribute::ON);
                osg::Uniform *var = new osg::Uniform(itTex2D->first.c_str(), texture);
                sset->addUniform(var);
                texture++;
            }
            std::map<std::string, osg::ref_ptr<osg::Texture3D> > tex3DUni = s->getTexture3DUniforms();
            std::map<std::string, osg::ref_ptr<osg::Texture3D> >::iterator itTex3D;
            for (itTex3D = tex3DUni.begin(); itTex3D != tex3DUni.end(); itTex3D++)
            {
                sset->setTextureAttributeAndModes(texture, itTex3D->second.get(), osg::StateAttribute::ON);
                osg::Uniform *var = new osg::Uniform(itTex3D->first.c_str(), texture);
                sset->addUniform(var);
                texture++;
            }
            std::map<std::string, osg::ref_ptr<osg::TextureCubeMap> > texCubeUni = s->getTextureCubeUniforms();
            std::map<std::string, osg::ref_ptr<osg::TextureCubeMap> >::iterator itTexCube;
            for (itTexCube = texCubeUni.begin(); itTexCube != texCubeUni.end(); itTexCube++)
            {
                sset->setTextureAttributeAndModes(texture, itTexCube->second.get(), osg::StateAttribute::ON);
                osg::Uniform *var = new osg::Uniform(itTexCube->first.c_str(), texture);
                sset->addUniform(var);
                texture++;
            }
        }

        if (appearance.material)
        {
            sset->setAttributeAndModes(appearance.material, osg::StateAttribute::ON);
        }

        if (appearance.color[3] < 0.99f)
        {
            sset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
            sset->setMode(GL_BLEND, osg::StateAttribute::ON);
        }
        else
        {
            sset->setRenderingHint(osg::StateSet::OPAQUE_BIN);
        }
    }

    n->setStateSet(sset);
}

bool AppearanceBehavior::buildFromXML(QDomElement *behaviorElement)
{
    // read all scopes
    QDomElement scopeElem = behaviorElement->firstChildElement("scope");
    while (!scopeElem.isNull())
    {
        Scope scope;
        scope.name = scopeElem.attribute("name", "").toStdString();
        scope.regexp = scopeElem.attribute("regexp", "");
        scope.geoNameSpace = scopeElem.attribute("namespace", "").toStdString();
        // color
        QDomElement colorElem = scopeElem.firstChildElement("color");
        if (!colorElem.isNull())
        {
            std::stringstream value;
            float val1, val2, val3, val4;
            value << colorElem.attribute("value", "0.0 0.0 0.0 0.0").toStdString();
            value >> val1 >> val2 >> val3 >> val4;
            osg::Vec4 color(val1, val2, val3, val4);
            scope.appearance.color = color;
            scope.appearance.material = createMaterial(color);
        }
        else
        {
            scope.appearance.material = NULL;
        }
        // shader
        QDomElement shaderElem = scopeElem.firstChildElement("shader");
        if (!shaderElem.isNull())
        {
            scope.appearance.shader = buildShaderFromXML(&shaderElem);
        }
        else
        {
            scope.appearance.shader = NULL;
        }
        _scopeList.push_back(scope);
        // next
        scopeElem = scopeElem.nextSiblingElement("scope");
    }
    return true;
}

osg::ref_ptr<osg::Material> AppearanceBehavior::createMaterial(osg::Vec4 color)
{
    osg::ref_ptr<osg::Material> material;
    material = new osg::Material();
    material->setColorMode(osg::Material::OFF);
    material->setAmbient(osg::Material::FRONT_AND_BACK, color * 0.6f);
    material->setDiffuse(osg::Material::FRONT_AND_BACK, color * 0.3f);
    material->setSpecular(osg::Material::FRONT_AND_BACK, color * 0.3f);
    material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    material->setAlpha(osg::Material::FRONT_AND_BACK, 1.0f);
    return material;
}

MyShader *AppearanceBehavior::buildShaderFromXML(QDomElement *shaderElement)
{
    // read a shader
    MyShader *shader = new MyShader();
    // file name of fragment shader
    QDomElement fragElem = shaderElement->firstChildElement("fragment");
    if (!fragElem.isNull())
    {
        std::string frag = fragElem.attribute("value", "test.frag").toStdString();
        shader->setFragmentFile(frag);
    }
    // file name of vertex shader
    QDomElement vertElem = shaderElement->firstChildElement("vertex");
    if (!vertElem.isNull())
    {
        std::string vert = vertElem.attribute("value", "test.vert").toStdString();
        shader->setVertexFile(vert);
    }
    // read uniform variables
    QDomElement varElem = shaderElement->firstChildElement("variable");
    while (!varElem.isNull())
    {
        // type
        std::string type = varElem.attribute("type", "int").toStdString();
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);
        // add to shader
        if (type == "int")
            shader->addUniform(varElem.attribute("name", "").toStdString(), varElem.attribute("value", "0").toInt());
        else if (type == "bool")
            shader->addUniform(varElem.attribute("name", "").toStdString(), bool(varElem.attribute("value", "0").toInt()));
        else if (type == "float")
        {
            std::string namef = varElem.attribute("name", "").toStdString();
            if (namef.compare("transparency") == 0 || namef.compare("transparent") == 0)
                shader->setTransparent(true);
            shader->addUniform(namef, varElem.attribute("value", "0.").toFloat());
        }
        else if (type == "vec2")
        {
            std::stringstream value;
            float val1, val2;
            value << varElem.attribute("value", "0. 0.").toStdString();
            value >> val1 >> val2;
            osg::Vec2 vec(val1, val2);
            shader->addUniform(varElem.attribute("name", "").toStdString(), vec);
        }
        else if (type == "vec3")
        {
            std::stringstream value;
            float val1, val2, val3;
            value << varElem.attribute("value", "0. 0. 0.").toStdString();
            value >> val1 >> val2 >> val3;
            osg::Vec3 vec(val1, val2, val3);
            shader->addUniform(varElem.attribute("name", "").toStdString(), vec);
        }
        else if (type == "vec4")
        {
            std::stringstream value;
            float val1, val2, val3, val4;
            value << varElem.attribute("value", "0. 0. 0. 0.").toStdString();
            value >> val1 >> val2 >> val3 >> val4;
            osg::Vec4 vec(val1, val2, val3, val4);
            shader->addUniform(varElem.attribute("name", "").toStdString(), vec);
        }
        else if (type == "texture1d")
        {
            std::string file = varElem.attribute("value", "").toStdString();
            osg::Image *image = new osg::Image();
            image = osgDB::readImageFile(file);
            osg::Texture1D *texture = new osg::Texture1D();
            texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
            texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
            texture->setWrap(osg::Texture::WRAP_R, osg::Texture::REPEAT);
            texture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
            texture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
            texture->setImage(0, image);
            shader->addUniform(varElem.attribute("name", "").toStdString(), texture);
        }
        else if (type == "texture2d")
        {
            std::string file = varElem.attribute("value", "").toStdString();
            osg::Image *image = new osg::Image();
            image = osgDB::readImageFile(file);
            //image->setFileName(file);
            osg::Texture2D *texture = new osg::Texture2D();
            texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
            texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
            texture->setWrap(osg::Texture::WRAP_R, osg::Texture::REPEAT);
            texture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
            texture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
            texture->setInternalFormat(image->getInternalTextureFormat()); // GL_RGBA32F_ARB);
            texture->setImage(0, image);
            shader->addUniform(varElem.attribute("name", "").toStdString(), texture);
        }
        else if (type == "texture3d")
        {
            std::string file = varElem.attribute("value", "").toStdString();
            osg::Image *image = new osg::Image();
            image = osgDB::readImageFile(file);
            osg::Texture3D *texture = new osg::Texture3D();
            texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
            texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
            texture->setWrap(osg::Texture::WRAP_R, osg::Texture::REPEAT);
            texture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
            texture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
            texture->setImage(0, image);
            shader->addUniform(varElem.attribute("name", "").toStdString(), texture);
        }
        else if (type == "texturecubemap")
        {
            osg::TextureCubeMap *texture = new osg::TextureCubeMap();
            QDomElement textElem = varElem.firstChildElement("texture");
            while (!textElem.isNull())
            {
                std::string file = textElem.attribute("file", "").toStdString();
                std::string textType = textElem.attribute("type", "").toStdString();
                osg::Image *image = new osg::Image();
                image = osgDB::readImageFile(file);
                if (textType.compare("left") == 0)
                    texture->setImage(osg::TextureCubeMap::NEGATIVE_X, image);
                else if (textType.compare("right") == 0)
                    texture->setImage(osg::TextureCubeMap::POSITIVE_X, image);
                else if (textType.compare("front") == 0)
                    texture->setImage(osg::TextureCubeMap::NEGATIVE_Z, image);
                else if (textType.compare("back") == 0)
                    texture->setImage(osg::TextureCubeMap::POSITIVE_Z, image);
                else if (textType.compare("top") == 0)
                    texture->setImage(osg::TextureCubeMap::POSITIVE_Y, image);
                else if (textType.compare("bottom") == 0)
                    texture->setImage(osg::TextureCubeMap::NEGATIVE_Y, image);
                textElem = textElem.nextSiblingElement();
            }
            shader->addUniform(varElem.attribute("name", "").toStdString(), texture);
        }
        // next variable
        varElem = varElem.nextSiblingElement();
    }

    return shader;
}
