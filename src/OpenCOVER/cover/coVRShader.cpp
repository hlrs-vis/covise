/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRShader.h"
#include "coVRFileManager.h"
#include "coVRPluginSupport.h"
#include "coVRSceneView.h"
#include <osg/Depth>
#include <config/CoviseConfig.h>
#include "VRSceneGraph.h"

#include <osg/Uniform>
#include <osg/Program>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Image>
#include <osg/Texture1D>
#include <osg/Texture2D>
#include <osg/Texture3D>
#include <osg/CullFace>
#include <osg/TextureCubeMap>
#include <osg/GL2Extensions>
#include <osgDB/ReadFile>
#include <osgUtil/TangentSpaceGenerator>
#include <util/coFileUtil.h>
#include <util/coTabletUIMessages.h>
#include <xercesc/dom/DOM.hpp>
#if XERCES_VERSION_MAJOR < 3
#include <xercesc/dom/DOMWriter.hpp>
#else
#include <xercesc/dom/DOMLSSerializer.hpp>
#endif
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <util/unixcompat.h>
#include <net/message.h>
#include <net/tokenbuffer.h>
#include <cstring>
#include <iostream>
#include <osg/io_utils>
using namespace opencover;
using namespace covise;

coVRUniform::coVRUniform(const coVRShader *s, const std::string &n, const std::string &t, const std::string &v)
{
    shader = s;
    name = n;
    type = t;
    value = v;
    overwrite = false;
    unique = false;
    if (type == "float")
    {
        float f = (float)strtod(value.c_str(), NULL);
        uniform = new osg::Uniform(name.c_str(), f);
    }
    else if (type == "int")
    {
        if (name == "Time")
        {
            uniform = coVRShaderList::instance()->getTime();
        }
        else if (name == "Stereo")
        {
            uniform = coVRShaderList::instance()->getStereo();
        }

        else if (name == "Duration")
        {
            uniform = coVRShaderList::instance()->getDuration();
        }
        else if (name == "ViewportWidth")
        {
            uniform = coVRShaderList::instance()->getViewportWidth();
        }
        else if (name == "ViewportHeight")
        {
            uniform = coVRShaderList::instance()->getViewportHeight();
        }
        else
        {
            int i = atoi(value.c_str());
            uniform = new osg::Uniform(name.c_str(), i);
        }
    }
    else if (type == "vec3")
    {
        float u = 0.0, v = 0.0, w = 0.0;
        sscanf(value.c_str(), "%f %f %f", &u, &v, &w);
        uniform = new osg::Uniform(name.c_str(), osg::Vec3(u, v, w));
    }
    else if (type == "vec2")
    {
        float u = 0.0, v = 0.0;
        sscanf(value.c_str(), "%f %f", &u, &v);
        uniform = new osg::Uniform(name.c_str(), osg::Vec2(u, v));
    }
    else if (type == "vec4")
    {
        float u = 0.0, v = 0.0, w = 0.0, a = 0.0;
        sscanf(value.c_str(), "%f %f %f %f", &u, &v, &w, &a);
        uniform = new osg::Uniform(name.c_str(), osg::Vec4(u, v, w, a));
    }
    else if (type == "sampler1D" || type == "sampler2D" || type == "sampler3D" || type == "samplerCube")
    {
        int texUnit = atoi(value.c_str());
        uniform = new osg::Uniform(name.c_str(), texUnit);
    }
    else if (type == "mat4")
    {
        osg::Matrix m;
        float values[16];
        //ab hier neu
        if (name == "Light")
        {
            uniform = coVRShaderList::instance()->getLight();
        }
        else
        {
            uniform = new osg::Uniform(name.c_str(), osg::Matrix::identity());
        }
        if (name == "Projection")
        {
            uniform = coVRShaderList::instance()->getProjection();
        }
        //bis hier neu
        if (strcasecmp(value.c_str(), "identity") == 0)
        {
            uniform->set(osg::Matrix::identity());
        }
        else
        {
            sscanf(value.c_str(), "%f %f %f %f  %f %f %f %f  %f %f %f %f  %f %f %f %f", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8], &values[9], &values[10], &values[11], &values[12], &values[13], &values[14], &values[15]);
            uniform->set(osg::Matrix(values));
        }
    }
}

osg::Texture::WrapMode coVRUniform::getWrapMode() const
{
    if (wrapMode.length() == 0)
        return osg::Texture::REPEAT;
    if (wrapMode == "CLAMP")
        return osg::Texture::CLAMP;
    return osg::Texture::REPEAT;
}

void coVRUniform::setTexture(const char *tf, int i)
{
    std::string fn = shader->findAsset(tf);
    if (!fn.empty())
    {
        textureFile = fn;
        osg::Image *image;
        image = osgDB::readImageFile(fn);
        if (image)
        {
            if (texture.get() == NULL)
            {
                if (type == "sampler3D")
                {
                    texture = new osg::Texture3D;
                    texture->setWrap(osg::Texture::WRAP_R, getWrapMode());
                    texture->setWrap(osg::Texture::WRAP_S, getWrapMode());
                    texture->setWrap(osg::Texture::WRAP_T, getWrapMode());
                }
                if (type == "sampler2D")
                {
                    texture = new osg::Texture2D;

                    texture->setWrap(osg::Texture::WRAP_R, getWrapMode());
                    texture->setWrap(osg::Texture::WRAP_S, getWrapMode());
                    texture->setWrap(osg::Texture::WRAP_T, getWrapMode());
                }
                if (type == "sampler1D")
                {
                    texture = new osg::Texture1D;

                    texture->setWrap(osg::Texture::WRAP_R, getWrapMode());
                    texture->setWrap(osg::Texture::WRAP_S, getWrapMode());
                    texture->setWrap(osg::Texture::WRAP_T, getWrapMode());
                }
                if (type == "samplerCube")
                {
                    texture = new osg::TextureCubeMap;
                }
            }
            texture->setImage(i, image);
            if (type == "samplerCube")
                cubeMapFiles[i] = fn;
        }
    }
}

void coVRUniform::setValue(osg::Matrix m)
{
    char ms[1600];
    sprintf(ms, "%f %f %f %f  %f %f %f %f  %f %f %f %f  %f %f %f %f", m(0, 0), m(0, 1), m(0, 2), m(0, 3), m(1, 0), m(1, 1), m(1, 2), m(1, 3), m(2, 0), m(2, 1), m(2, 2), m(2, 3), m(3, 0), m(3, 1), m(3, 2), m(3, 3));
    value = ms;
    uniform->set(m);
}

void coVRUniform::setValue(float f)
{
    char fs[100];
    sprintf(fs, "%f", f);
    value = fs;
    uniform->set(f);
}

void coVRUniform::setValue(osg::Vec3 v)
{
    char vs[300];
    sprintf(vs, "%f %f %f", v[0], v[1], v[2]);
    value = vs;
    uniform->set(v);
}

void coVRUniform::setValue(osg::Vec4 v)
{
    char vs[400];
    sprintf(vs, "%f %f %f %f", v[0], v[1], v[2], v[3]);
    value = vs;
    uniform->set(v);
}
void coVRUniform::setWrapMode(std::string wm)
{
    wrapMode = wm;
    if (texture.valid())
    {
        texture->setWrap(osg::Texture::WRAP_R, getWrapMode());
        texture->setWrap(osg::Texture::WRAP_S, getWrapMode());
        texture->setWrap(osg::Texture::WRAP_T, getWrapMode());
    }
}

void coVRUniform::setValue(const char *val)
{
    value = val;
    if (type == "float")
    {

        float f = (float)strtod(val, NULL);
        uniform->set(f);
    }
    else if (type == "int")
    {
        int i = atoi(val);
        uniform->set(i);
    }
    else if (type == "vec3")
    {
        float u = 0.0, v = 0.0, w = 0.0;
        sscanf(val, "%f %f %f", &u, &v, &w);
        uniform->set(osg::Vec3(u, v, w));
    }
    else if (type == "vec2")
    {
        float u = 0.0, v = 0.0;
        sscanf(val, "%f %f", &u, &v);
        uniform->set(osg::Vec2(u, v));
    }
    else if (type == "vec4")
    {
        float u = 0.0, v = 0.0, w = 0.0, a = 0.0;
        sscanf(val, "%f %f %f %f", &u, &v, &w, &a);
        uniform->set(osg::Vec4(u, v, w, a));
    }
    else if (type == "mat4")
    {
        float values[16];
        if (strcasecmp(val, "identity") == 0)
        {
            uniform = new osg::Uniform(name.c_str(), osg::Matrix::identity());
        }
        else
        {
            sscanf(val, "%f %f %f %f  %f %f %f %f  %f %f %f %f  %f %f %f %f", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8], &values[9], &values[10], &values[11], &values[12], &values[13], &values[14], &values[15]);
            uniform = new osg::Uniform(name.c_str(), osg::Matrix(values));
        }
    }
    else if (type == "sampler2D" || type == "sampler1D" || type == "sampler3D" || type == "samplerCube")
    {
        int i = atoi(val);
        uniform->set(i);
    }
}

coVRUniform::~coVRUniform()
{
}

coVRAttribute::coVRAttribute(const std::string &n, const std::string &t, const std::string &v)
{
    name = n;
    type = t;
    value = v;
}

coVRAttribute::~coVRAttribute()
{
}

coVRShader::coVRShader(const std::string &n, const std::string &d)
{
    name = n;
    dir = d;
    geometryShader = NULL;
    geomParams[0] = 3;
    geomParams[1] = GL_POINTS;
    geomParams[2] = GL_POINTS;
    transparent = false;
    cullFace = -1;
    opaque = false;
    if (name.rfind(".xml") == (name.length() - 4))
        name = std::string(name, 0, name.length() - 4);
    coVRShaderList::instance()->push_back(this);
    loadMaterial();
}

std::string coVRShader::findAsset(const std::string &path) const
{
    if (path.empty())
        return "";
    const char *fn = NULL;
    if (path[0] == '/')
    {
        fn = coVRFileManager::instance()->getName(path.c_str());
        if (!fn)
            return "";
        return fn;
    }
    fn = coVRFileManager::instance()->getName((dir + "/" + path).c_str());
    if (fn)
        return fn;
    fn = coVRFileManager::instance()->getName(path.c_str());
    if (fn)
        return fn;

    return "";
}

void coVRShader::loadMaterial()
{
    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);
    const char *fn = NULL;
    if (dir.length() != 0)
    {
        std::string buf = dir + "/" + name + ".xml";
        fn = coVRFileManager::instance()->getName(buf.c_str());
    }
    if (fn == NULL)
    {
        std::string buf = "share/covise/materials/" + name + ".xml";
        fn = coVRFileManager::instance()->getName(buf.c_str());
    }
    if (fn == NULL)
    {
        std::string buf = name + ".xml";
        fn = coVRFileManager::instance()->getName(buf.c_str());
    }
    if (fn)
    {
        fileName = fn;
        std::string::size_type lastslash = fileName.rfind('/');
        if (lastslash == 0)
        {
            dir = "/";
        }
        else if (lastslash != std::string::npos)
        {
            dir = fileName.substr(0, lastslash);
        }
        else
        {
            dir = "";
        }
        try
        {
            parser->parse(fn);
        }
        catch (...)
        {
            cerr << "error parsing Material file" << fileName << endl;
        }
    }
    else
    {
        cerr << "could not find Material " << name << endl;
    }

    xercesc::DOMDocument *xmlDoc = parser->getDocument();
    xercesc::DOMElement *rootElement = NULL;
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    if (rootElement)
    {
        const char *transp = xercesc::XMLString::transcode(rootElement->getAttribute(xercesc::XMLString::transcode("transparent")));
        if (transp && strcasecmp(transp, "true") == 0)
        {
            transparent = true;
        }

        const char *op = xercesc::XMLString::transcode(rootElement->getAttribute(xercesc::XMLString::transcode("opaque")));
        if (op && strcasecmp(op, "true") == 0)
        {
            opaque = true;
        }

        const char *cullString = xercesc::XMLString::transcode(rootElement->getAttribute(xercesc::XMLString::transcode("cullFace")));
        if (cullString)
        {
            if (strcasecmp(cullString, "true") == 0)
            {
                cullFace = osg::CullFace::BACK;
            }
            else if (strcasecmp(cullString, "back") == 0)
            {
                cullFace = osg::CullFace::BACK;
            }
            else if (strcasecmp(cullString, "front") == 0)
            {
                cullFace = osg::CullFace::FRONT;
            }
            else if (strcasecmp(cullString, "front_and_back") == 0)
            {
                cullFace = osg::CullFace::FRONT_AND_BACK;
            }
            else if (strcasecmp(cullString, "none") == 0 || strcasecmp(cullString, "off") == 0)
            {
                cullFace = 0;
            }
        }
        xercesc::DOMNodeList *nodeList = rootElement->getChildNodes();
        for (int i = 0; i < nodeList->getLength(); ++i)
        {
            xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
            if (!node)
                continue;
            const char *tagName = xercesc::XMLString::transcode(node->getTagName());
            if (tagName)
            {
                if (strcmp(tagName, "attribute") == 0)
                {
                    const char *type = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("type")));
                    const char *value = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("value")));
                    const char *attributeName = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("name")));
                    if (type != NULL && value != NULL && attributeName != NULL)
                        attributes.push_back(new coVRAttribute(attributeName, type, value));
                }
                else if (strcmp(tagName, "uniform") == 0)
                {
                    const char *type = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("type")));
                    const char *value = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("value")));
                    const char *uniformName = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("name")));
                    const char *minValue = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("min")));
                    const char *maxValue = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("max")));
                    const char *textureName = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("texture")));
                    const char *texture1Name = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("texture1")));
                    const char *overwrite = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("overwrite")));
                    const char *unique = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("unique")));
                    const char *wm = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("wrapMode")));
                    if (type != NULL && value != NULL && uniformName != NULL)
                    {
                        coVRUniform *u = new coVRUniform(this, uniformName, type, value);
                        uniforms.push_back(u);
                        if (minValue && minValue[0] != '\0')
                            u->setMin(minValue);
                        if (maxValue && maxValue[0] != '\0')
                            u->setMax(maxValue);
                        u->setOverwrite(overwrite && strcmp(overwrite, "true") == 0);
                        u->setUnique(unique && strcmp(unique, "true") == 0);
                        if (wm)
                        {
                            u->setWrapMode(wm);
                        }
                        if (textureName && textureName[0] != '\0')
                        {
                            u->setTexture(textureName);
                        }
                        if (texture1Name && texture1Name[0] != '\0')
                        {
                            for (int i = 0; i < 6; i++)
                            {
                                char attrName[100];
                                sprintf(attrName, "texture%d", i + 1);
                                const char *textureName = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode(attrName)));
                                u->setTexture(textureName, i);
                            }
                        }
                    }
                }
                else if (strcmp(tagName, "fragmentProgram") == 0)
                {
                    std::string code = "";
                    const char *value = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("value")));
                    if (value && value[0] != '\0')
                    {
                        std::string filename = findAsset(value);
                        if (!filename.empty())
                        {
                            std::ifstream t(filename);
                            std::stringstream buffer;
                            buffer << t.rdbuf();
                            code = buffer.str();
                        }
                    }
                    if (code == "")
                    {
                        const char *c = xercesc::XMLString::transcode(node->getTextContent());
                        if (c && c[0] != '\0')
                            code = c;
                    }
                    if (code != "")
                    {
                        fragmentShader = new osg::Shader(osg::Shader::FRAGMENT, code);
                    }
                }
                else if (strcmp(tagName, "geometryProgram") == 0)
                {
                    const char *numVertices = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("numVertices")));
                    geomParams[0] = atoi(numVertices);
                    //FIXME glGetIntegerv requires a valid OpenGL context, otherwise: crash
                    //if (geomParams[0] != 0) glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT,&geomParams[0]);
                    if (geomParams[0] != 0)
                        geomParams[0] = 1024;

                    const char *inputType = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("inputType")));
                    if (strcmp(inputType, "POINTS") == 0)
                        geomParams[1] = GL_POINTS;
                    else if (strcmp(inputType, "LINES") == 0)
                        geomParams[1] = GL_LINES;
                    else if (strcmp(inputType, "LINES_ADJACENCY_EXT") == 0)
                        geomParams[1] = GL_LINES_ADJACENCY_EXT;
                    else if (strcmp(inputType, "TRIANGLES_ADJACENCY_EXT") == 0)
                        geomParams[1] = GL_TRIANGLES_ADJACENCY_EXT;
                    else
                        geomParams[1] = GL_TRIANGLES;

                    const char *outputType = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("outputType")));
                    if (strcmp(outputType, "POINTS") == 0)
                        geomParams[2] = GL_POINTS;
                    else if (strcmp(outputType, "LINE_STRIP") == 0)
                        geomParams[2] = GL_LINE_STRIP;
                    else
                        geomParams[2] = GL_TRIANGLE_STRIP;

                    const char *code = xercesc::XMLString::transcode(node->getTextContent());
                    if (code && code[0] != '\0')
                        geometryShader = new osg::Shader(osg::Shader::GEOMETRY, code);
                }
                else if (strcmp(tagName, "vertexProgram") == 0)
                {
                    std::string code = "";
                    const char *value = xercesc::XMLString::transcode(node->getAttribute(xercesc::XMLString::transcode("value")));
                    if (value && value[0] != '\0')
                    {
                        std::string filename = findAsset(value);
                        if (!filename.empty())
                        {
                            std::ifstream t(filename);
                            std::stringstream buffer;
                            buffer << t.rdbuf();
                            code = buffer.str();
                        }
                    }
                    if (code == "")
                    {
                        const char *c = xercesc::XMLString::transcode(node->getTextContent());
                        if (c && c[0] != '\0')
                            code = c;
                    }
                    if (code != "")
                    {
                        vertexShader = new osg::Shader(osg::Shader::VERTEX, code);
                    }
                }
                else if (strcmp(tagName, "tessControlProgram") == 0)
                {
                    const char *code = xercesc::XMLString::transcode(node->getTextContent());
                    if (code && code[0] != '\0')
                        tessControlShader = new osg::Shader(osg::Shader::TESSCONTROL, code);
                }
                else if (strcmp(tagName, "tessEvalProgram") == 0)
                {
                    const char *code = xercesc::XMLString::transcode(node->getTextContent());
                    if (code && code[0] != '\0')
                        tessEvalShader = new osg::Shader(osg::Shader::TESSEVALUATION, code);
                }
            }
        }
    }
}

void coVRShader::setUniformesFromAttribute(const char *uniformValues)
{
    char *tmpStr = new char[strlen(uniformValues) + 1];
    strcpy(tmpStr, uniformValues);
    char *c = tmpStr;
    while (*c != '\0')
    {
        while (*c == ' ' && *c != '\0')
            c++;
        if (*c == '\0')
            break;
        char *name = c;
        c++;
        while (*c != '=' && *c != '\0')
            c++;
        if (*c == '\0')
            break;
        *c = '\0';
        c++;
        while (*c != '"' && *c != '\0')
            c++;
        if (*c == '\0')
            break;
        c++;
        char *value = c;
        while (*c != '"' && *c != '\0')
            c++;
        if (*c == '\0')
            break;
        *c = '\0';
        c++;
        std::list<coVRUniform *>::iterator it;
        for (it = uniforms.begin(); it != uniforms.end(); it++)
        {
            if ((*it)->getName() == name)
            {
                (*it)->setValue(value);
            }
        }
    }
}

osg::Uniform *coVRShader::getUniform(const std::string &name)
{
    std::list<coVRUniform *>::iterator it;
    for (it = uniforms.begin(); it != uniforms.end(); it++)
    {
        if ((*it)->getName() == name)
        {
            return (*it)->uniform.get();
        }
    }
    return NULL;
}

void coVRShader::setMatrixUniform(const std::string &name, osg::Matrix m)
{
    std::list<coVRUniform *>::iterator it;
    for (it = uniforms.begin(); it != uniforms.end(); it++)
    {
        if ((*it)->getName() == name)
        {
            (*it)->setValue(m);
        }
    }
}

void coVRShader::setFloatUniform(const std::string &name, float f)
{
    std::list<coVRUniform *>::iterator it;
    for (it = uniforms.begin(); it != uniforms.end(); it++)
    {
        if ((*it)->getName() == name)
        {
            (*it)->setValue(f);
        }
    }
}

void coVRShader::setVec3Uniform(const std::string &name, osg::Vec3 v)
{
    std::list<coVRUniform *>::iterator it;
    for (it = uniforms.begin(); it != uniforms.end(); it++)
    {
        if ((*it)->getName() == name)
        {
            (*it)->setValue(v);
        }
    }
}

void coVRShader::setVec4Uniform(const std::string &name, osg::Vec4 v)
{
    std::list<coVRUniform *>::iterator it;
    for (it = uniforms.begin(); it != uniforms.end(); it++)
    {
        if ((*it)->getName() == name)
        {
            (*it)->setValue(v);
        }
    }
}

void coVRShader::setData(TokenBuffer &tb)
{
    int type;
    tb >> type;
    if (type == SHADER_UNIFORM)
    {
        //const char *name;
        //const char *value;
        std::string name;
        std::string value;
        std::string textureFile;
        tb >> name;
        tb >> value;
        tb >> textureFile;
        std::list<coVRUniform *>::iterator it;
        for (it = uniforms.begin(); it != uniforms.end(); it++)
        {
            if ((*it)->getName() == name)
            {
                (*it)->setValue(value.c_str());
                (*it)->setTexture(textureFile.c_str());
            }
        }
    }
    else if (type == SHADER_FRAGMENT)
    {
        std::string code;
        tb >> code;
        fragmentShader->setShaderSource(code.c_str());
        fragmentShader->dirtyShader();
    }
    else if (type == SHADER_GEOMETRY)
    {
        std::string code;
        tb >> code;
        if (geometryShader == NULL)
        {
            geometryShader = new osg::Shader(osg::Shader::GEOMETRY, code);

            program->addShader(geometryShader.get());
            program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, geomParams[0]);
            program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, geomParams[1]);
            program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, geomParams[2]);
        }
        else
        {
            geometryShader->setShaderSource(code.c_str());
        }
        geometryShader->dirtyShader();
    }
    else if (type == SHADER_VERTEX)
    {
        std::string code;
        tb >> code;
        vertexShader->setShaderSource(code.c_str());
        vertexShader->dirtyShader();
    }
    else if (type == SHADER_TESSCONTROL)
    {
        std::string code;
        tb >> code;
        tessControlShader->setShaderSource(code.c_str());
        tessControlShader->dirtyShader();
    }
    else if (type == SHADER_TESSEVAL)
    {
        std::string code;
        tb >> code;
        tessEvalShader->setShaderSource(code.c_str());
        tessEvalShader->dirtyShader();
    }
}

void coVRShader::storeMaterial()
{
    xercesc::DOMImplementation *impl;
    impl = xercesc::DOMImplementationRegistry::getDOMImplementation(xercesc::XMLString::transcode("Core"));

    std::string ShaderName = name;
    for (int i = 0; i < ShaderName.length(); i++)
    {
        if (ShaderName[i] == ' ')
            ShaderName[i] = '_';
    }
    xercesc::DOMDocument *document = impl->createDocument(0, xercesc::XMLString::transcode(ShaderName.c_str()), 0);

    xercesc::DOMElement *rootElement = document->getDocumentElement();
    if (transparent)
        rootElement->setAttribute(xercesc::XMLString::transcode("transparent"), xercesc::XMLString::transcode("true"));
    if (opaque)
        rootElement->setAttribute(xercesc::XMLString::transcode("opaque"), xercesc::XMLString::transcode("true"));
    if (cullFace == osg::CullFace::BACK)
        rootElement->setAttribute(xercesc::XMLString::transcode("cullFace"), xercesc::XMLString::transcode("back"));
    if (cullFace == osg::CullFace::FRONT)
        rootElement->setAttribute(xercesc::XMLString::transcode("cullFace"), xercesc::XMLString::transcode("front"));
    if (cullFace == osg::CullFace::FRONT_AND_BACK)
        rootElement->setAttribute(xercesc::XMLString::transcode("cullFace"), xercesc::XMLString::transcode("front_and_back"));
    if (cullFace == 0)
        rootElement->setAttribute(xercesc::XMLString::transcode("cullFace"), xercesc::XMLString::transcode("off"));

    for (std::list<coVRUniform *>::iterator it = uniforms.begin(); it != uniforms.end(); ++it)
    {
        xercesc::DOMElement *uniform = document->createElement(xercesc::XMLString::transcode("uniform"));
        uniform->setAttribute(xercesc::XMLString::transcode("name"), xercesc::XMLString::transcode((*it)->getName().c_str()));
        uniform->setAttribute(xercesc::XMLString::transcode("type"), xercesc::XMLString::transcode((*it)->getType().c_str()));
        uniform->setAttribute(xercesc::XMLString::transcode("value"), xercesc::XMLString::transcode((*it)->getValue().c_str()));
        const std::string *fn = (*it)->getCubeMapFiles();
        if (fn[0].empty())
        {
            if (!(*it)->getTextureFileName().empty())
                uniform->setAttribute(xercesc::XMLString::transcode("texture"), xercesc::XMLString::transcode((*it)->getTextureFileName().c_str()));
        }
        else
        {
            for (int i = 0; i < 6; i++)
            {
                char attrName[100];
                sprintf(attrName, "texture%d", i + 1);
                uniform->setAttribute(xercesc::XMLString::transcode(attrName), xercesc::XMLString::transcode(fn[i].c_str()));
            }
        }

        if (!(*it)->getMin().empty())
        {
            uniform->setAttribute(xercesc::XMLString::transcode("min"), xercesc::XMLString::transcode((*it)->getMin().c_str()));
            uniform->setAttribute(xercesc::XMLString::transcode("max"), xercesc::XMLString::transcode((*it)->getMax().c_str()));
        }
        if ((*it)->isUnique())
        {
            uniform->setAttribute(xercesc::XMLString::transcode("unique"), xercesc::XMLString::transcode("true"));
        }
        if ((*it)->doOverwrite())
        {
            uniform->setAttribute(xercesc::XMLString::transcode("overwrite"), xercesc::XMLString::transcode("true"));
        }
        rootElement->appendChild(uniform);
    }

    for (std::list<coVRAttribute *>::iterator it = attributes.begin(); it != attributes.end(); ++it)
    {
        xercesc::DOMElement *attrib = document->createElement(xercesc::XMLString::transcode("attribute"));
        attrib->setAttribute(xercesc::XMLString::transcode("name"), xercesc::XMLString::transcode((*it)->getName().c_str()));
        attrib->setAttribute(xercesc::XMLString::transcode("type"), xercesc::XMLString::transcode((*it)->getType().c_str()));
        attrib->setAttribute(xercesc::XMLString::transcode("value"), xercesc::XMLString::transcode((*it)->getValue().c_str()));
        rootElement->appendChild(attrib);
    }
    if (vertexShader.get() != NULL)
    {
        xercesc::DOMElement *vertexProgram = document->createElement(xercesc::XMLString::transcode("vertexProgram"));
        vertexProgram->setTextContent(xercesc::XMLString::transcode(vertexShader->getShaderSource().c_str()));
        rootElement->appendChild(vertexProgram);
    }
    if (tessControlShader.get() != NULL)
    {
        xercesc::DOMElement *vertexProgram = document->createElement(xercesc::XMLString::transcode("tessControlProgram"));
        vertexProgram->setTextContent(xercesc::XMLString::transcode(tessControlShader->getShaderSource().c_str()));
        rootElement->appendChild(vertexProgram);
    }
    if (tessEvalShader.get() != NULL)
    {
        xercesc::DOMElement *vertexProgram = document->createElement(xercesc::XMLString::transcode("tessEvalProgram"));
        vertexProgram->setTextContent(xercesc::XMLString::transcode(tessEvalShader->getShaderSource().c_str()));
        rootElement->appendChild(vertexProgram);
    }
    if (geometryShader.get() != NULL)
    {
        xercesc::DOMElement *geometryProgram = document->createElement(xercesc::XMLString::transcode("geometryProgram"));
        char numVertices[GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT];
        sprintf(numVertices, "%d", geomParams[0]);
        geometryProgram->setAttribute(xercesc::XMLString::transcode("numVertices"), xercesc::XMLString::transcode(numVertices));
        //    GLint geometryProgram->getParamenter(GL_GEOMETRY_INPUT_TYPE_EXT);

        std::string inputType;
        switch (geomParams[1])
        {
        case GL_TRIANGLES:
            inputType = "TRIANGLES";
            break;
        case GL_POINTS:
            inputType = "POINTS";
            break;
        case GL_LINES:
            inputType = "LINES";
            break;
        case GL_LINES_ADJACENCY_EXT:
            inputType = "LINES_ADJACENCY_EXT";
            break;
        case GL_TRIANGLES_ADJACENCY_EXT:
            inputType = "TRIANGLES_ADJACENCY_EXT";
            break;
        }

        geometryProgram->setAttribute(xercesc::XMLString::transcode("inputType"), xercesc::XMLString::transcode(inputType.c_str()));

        std::string outputType;
        switch (geomParams[2])
        {
        case GL_TRIANGLE_STRIP:
            outputType = "TRIANGLE_STRIP";
            break;
        case GL_POINTS:
            outputType = "POINTS";
            break;
        case GL_LINES:
            outputType = "LINES";
            break;
        case GL_LINE_STRIP:
            inputType = "LINE_STRIP";
            break;
        }

        geometryProgram->setAttribute(xercesc::XMLString::transcode("outputType"), xercesc::XMLString::transcode(outputType.c_str()));
        geometryProgram->setTextContent(xercesc::XMLString::transcode(geometryShader->getShaderSource().c_str()));
        rootElement->appendChild(geometryProgram);
    }
    if (fragmentShader.get() != NULL)
    {
        xercesc::DOMElement *fragmentProgram = document->createElement(xercesc::XMLString::transcode("fragmentProgram"));
        fragmentProgram->setTextContent(xercesc::XMLString::transcode(fragmentShader->getShaderSource().c_str()));
        rootElement->appendChild(fragmentProgram);
    }

#if XERCES_VERSION_MAJOR < 3
    xercesc::DOMWriter *writer = impl->createDOMWriter();
    // set the format-pretty-print feature
    if (writer->canSetFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
        writer->setFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);

    xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget(fileName.c_str());
    bool written = writer->writeNode(xmlTarget, *rootElement);
    if (!written)
        fprintf(stderr, "Material::save info: Could not open file for writing %s!\n", fileName.c_str());

    delete writer;
    delete xmlTarget;
#else

    xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();
    //xercesc::DOMConfiguration* dc = writer->getDomConfig();
    //dc->setParameter(xercesc::XMLUni::fgDOMErrorHandler,errorHandler);
    //dc->setParameter(xercesc::XMLUni::fgDOMWRTDiscardDefaultContent,true);

    xercesc::DOMLSOutput *theOutput = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
    theOutput->setEncoding(xercesc::XMLString::transcode("utf8"));

    bool written = writer->writeToURI(rootElement, xercesc::XMLString::transcode(fileName.c_str()));
    if (!written)
        fprintf(stderr, "Material::save info: Could not open file for writing %s!\n", fileName.c_str());
    delete writer;

#endif
    delete document;
}

coVRShaderInstance *coVRShader::apply(osg::Node *node)
{
    coVRShaderInstance *lastInstance = NULL;

    coVRShaderList::instance()->remove(node); // remove all old shaders
    osg::Geode *geode = dynamic_cast<osg::Geode *>(node);
    osg::Group *group = dynamic_cast<osg::Group *>(node);
    if (geode)
    {
        for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
        {
            osg::Drawable *drawable = geode->getDrawable(i);
            if (drawable)
            {
                lastInstance = apply(geode, drawable);
            }
        }
    }
    else if (group)
    {
        for (unsigned int i = 0; i < group->getNumChildren(); i++)
        {
            lastInstance = apply(group->getChild(i));
        }
    }
    else
    {
        osg::StateSet *st = node->getOrCreateStateSet();
        if (st)
        {
            apply(st);
        }
    }
    return lastInstance;
}

void coVRShaderList::remove(osg::Node *node)
{
    osg::Geode *geode = dynamic_cast<osg::Geode *>(node);
    osg::Group *group = dynamic_cast<osg::Group *>(node);
    if (geode)
    {

        for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
        {

            osg::StateSet *stateset = geode->getOrCreateStateSet();
            //remove all uniforms
            osg::StateSet::UniformList ul = stateset->getUniformList();

            if (ul.empty())
            {
                stateset = geode->getDrawable(i)->getOrCreateStateSet();
                ul = stateset->getUniformList();
            }

            int numStages = stateset->getTextureAttributeList().size();
            if (numStages > 0)
                for (int i = 0; i < numStages; i++)
                {
                    osg::StateAttribute *stateTexture = stateset->getTextureAttribute(i, osg::StateAttribute::TEXTURE);
                    if (stateTexture != NULL)
                    {
                        coVRShaderList::iterator shaderIt = begin();
                        do
                        {
                            std::list<coVRUniform *> uniformList = (*shaderIt)->getUniforms();
                            std::list<coVRUniform *>::iterator it;
                            for (it = uniformList.begin(); it != uniformList.end(); it++)
                                if ((*it)->texture.get() == stateTexture)
                                {
                                    stateset->removeTextureAttribute(i, stateTexture);
                                    break;
                                }
                            shaderIt++;
                        } while ((shaderIt != end()) && stateset->getTextureAttribute(i, osg::StateAttribute::TEXTURE));
                    }
                }

            ul.clear();
            // remove textures
            for (int i = 1; i < 8; ++i)
            {
                if (stateset->getTextureAttribute(i, osg::StateAttribute::TEXTURE) != NULL)
                {
                    stateset->setTextureMode(i, GL_TEXTURE_1D, osg::StateAttribute::OFF);
                    stateset->setTextureMode(i, GL_TEXTURE_2D, osg::StateAttribute::OFF);
                    stateset->setTextureMode(i, GL_TEXTURE_3D, osg::StateAttribute::OFF);
                    stateset->setTextureMode(i, GL_TEXTURE_CUBE_MAP, osg::StateAttribute::OFF);
                }
            }
            //remove shaders
            while (stateset->getAttribute(osg::StateAttribute::PROGRAM) != NULL)
                stateset->removeAttribute(stateset->getAttribute(osg::StateAttribute::PROGRAM));

            /*        std::list<coVRShaderInstance *>::iterator it;
         for(it=instances.begin();it != instances.end(); it++)
         {
            if((*it)->drawable == drawable)
            {
               coVRShaderInstance *si = *it;
               instances.erase(it);
               delete si;
               break;
            }
         }*/
        }
    }
    else if (group)
    {
        for (unsigned int i = 0; i < group->getNumChildren(); i++)
        {
            remove(group->getChild(i));
        }
    }
    else
    {
        osg::StateSet *st = node->getStateSet();
        if (st)
        {

            while (st->getAttribute(osg::StateAttribute::PROGRAM) != NULL)
                st->removeAttribute(st->getAttribute(osg::StateAttribute::PROGRAM));
        }
    }
}

void coVRShader::apply(osg::StateSet *stateset)
{
    if (!stateset)
        return;

    if (transparent)
    {
        stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
        stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        stateset->setNestRenderBins(false);
    }
    if (program.get() == NULL)
    {
        program = new osg::Program;
        if (fragmentShader.get())
            program->addShader(fragmentShader.get());
        if (geometryShader.get())
        {
            program->addShader(geometryShader.get());
            program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, geomParams[0]);
            program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, geomParams[1]);
            program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, geomParams[2]);
        }
        if (vertexShader.get())
            program->addShader(vertexShader.get());
        if (tessControlShader.get())
            program->addShader(tessControlShader.get());
        if (tessEvalShader.get())
            program->addShader(tessEvalShader.get());

        std::list<coVRAttribute *>::iterator it;
        for (it = attributes.begin(); it != attributes.end(); it++)
        {
            coVRAttribute *a = *it;
            int attributeNumber = atoi(a->getValue().c_str());
            program->addBindAttribLocation(a->getName(), attributeNumber);
        }
    }
    stateset->setAttributeAndModes(program.get(), osg::StateAttribute::ON);

    std::list<coVRUniform *>::iterator it;
    for (it = uniforms.begin(); it != uniforms.end(); it++)
    {
        stateset->addUniform((*it)->uniform.get());
        if ((*it)->texture.get()) // if we have a texture
        {
            int stage = 0;
            // set texture only, if not already available or if overwrite is set
            sscanf((*it)->getValue().c_str(), "%d", &stage);
            if (stateset->getTextureAttribute(stage, osg::StateAttribute::TEXTURE) == NULL || (*it)->doOverwrite())
            {
                stateset->setTextureAttributeAndModes(stage, (*it)->texture.get(), osg::StateAttribute::ON);
            }
        }
    }
}

coVRShaderInstance *coVRShader::apply(osg::Geode *geode, osg::Drawable *drawable)
{
    coVRShaderInstance *instance = NULL;
    osg::Geometry *geo = dynamic_cast<osg::Geometry *>(drawable);
    if (geo)
    {
        osg::StateSet *stateset = geode->getOrCreateStateSet();

        if (transparent)
        {
            osg::StateSet *drawableSS = geo->getOrCreateStateSet();
            drawableSS->setMode(GL_BLEND, osg::StateAttribute::ON);
            drawableSS->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
            // Disable writing to depth buffer.
            osg::Depth *depth = new osg::Depth;
            depth->setWriteMask(false);
            drawable->getOrCreateStateSet()->setAttributeAndModes(depth, osg::StateAttribute::ON);
        }
        if (opaque)
        {
            osg::StateSet *drawableSS = geo->getOrCreateStateSet();
            drawableSS->setMode(GL_BLEND, osg::StateAttribute::OFF);
            drawableSS->setRenderingHint(osg::StateSet::OPAQUE_BIN);
            // Enable writing to depth buffer.
            osg::Depth *depth = new osg::Depth;
            depth->setWriteMask(true);
            drawable->getOrCreateStateSet()->setAttributeAndModes(depth, osg::StateAttribute::ON);
            stateset->setNestRenderBins(false);
        }
        if (cullFace != -1)
        {
            if (cullFace != 0)
            {
                osg::CullFace *cF = new osg::CullFace();
                cF->setMode((osg::CullFace::Mode)cullFace);
                stateset->setAttributeAndModes(cF, osg::StateAttribute::ON);
            }
            else
            {
                osg::CullFace *cF = new osg::CullFace();
                stateset->setAttributeAndModes(cF, osg::StateAttribute::OFF);
            }
        }
        if (program.get() == NULL)
        {
            program = new osg::Program;
            if (fragmentShader.get())
                program->addShader(fragmentShader.get());
            if (geometryShader.get())
            {
                program->addShader(geometryShader.get());
                program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, geomParams[0]);
                program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, geomParams[1]);
                program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, geomParams[2]);
            }
            if (vertexShader.get())
                program->addShader(vertexShader.get());
            if (tessControlShader.get())
                program->addShader(tessControlShader.get());
            if (tessEvalShader.get())
                program->addShader(tessEvalShader.get());

            std::list<coVRAttribute *>::iterator it;
            for (it = attributes.begin(); it != attributes.end(); it++)
            {
                coVRAttribute *a = *it;
                int attributeNumber = atoi(a->getValue().c_str());
                program->addBindAttribLocation(a->getName(), attributeNumber);
            }
        }
        stateset->setAttributeAndModes(program.get(), osg::StateAttribute::ON);

        std::list<coVRUniform *>::iterator it;
        for (it = uniforms.begin(); it != uniforms.end(); it++)
        {
            if ((*it)->isUnique())
            {
                if (instance == NULL)
                    instance = new coVRShaderInstance(drawable);
                instance->addUniform(*((*it)->uniform.get()));
            }
            else
            {
                stateset->addUniform((*it)->uniform.get());
            }
            if ((*it)->texture.get()) // if we have a texture
            {
                int stage = 0;
                // set texture only, if not already available or if overwrite is set
                sscanf((*it)->getValue().c_str(), "%d", &stage);
                if (stateset->getTextureAttribute(stage, osg::StateAttribute::TEXTURE) == NULL || (*it)->doOverwrite())
                {
                    stateset->setTextureAttributeAndModes(stage, (*it)->texture.get(), osg::StateAttribute::ON);
                }
            }
        }
        if (instance != NULL)
        {
            std::list<osg::ref_ptr<osg::Uniform> >::iterator it;
            for (it = instance->getUniforms().begin(); it != instance->getUniforms().end(); it++)
            {
                stateset->addUniform((*it).get());
            }
            instances.push_back(instance);
        }
        bool needTangent = false;
        std::list<coVRAttribute *>::iterator ait;
        for (ait = attributes.begin(); ait != attributes.end(); ait++)
        {
            coVRAttribute *a = *ait;
            int attributeNumber = atoi(a->getValue().c_str());
            if ((a->getType() == "tangent" || a->getType() == "binormal" || a->getType() == "normal") && geo->getVertexAttribArray(attributeNumber) == NULL)
            {
                needTangent = true;
                break;
            }
        }
        if (needTangent)
        {
            osg::ref_ptr<osgUtil::TangentSpaceGenerator> tsg = new osgUtil::TangentSpaceGenerator;
            int normalUnit = 0;
            std::list<coVRUniform *>::iterator it;
            // find a sampler which has normal in its name and take this texture unit to create tangents and binormals
            for (it = uniforms.begin(); it != uniforms.end(); it++)
            {
                coVRUniform *u = (*it);
                if (u->getType() == "sampler2D")
                {
                    if (u->getName().find("ormal") != string::npos)
                    {
                        int unit = atoi(u->getValue().c_str());
                        const osg::Array *tx = geo->getTexCoordArray(unit);
                        if (tx != NULL)
                        {
                            normalUnit = unit;
                            break;
                        }
                    }
                }
            }
            tsg->generate(geo, normalUnit);
            if (!tsg->getTangentArray()->empty())
            {
                std::list<coVRAttribute *>::iterator ait;
                for (ait = attributes.begin(); ait != attributes.end(); ait++)
                {
                    coVRAttribute *a = *ait;
                    int attributeNumber = atoi(a->getValue().c_str());
                    if (geo->getVertexAttribArray(attributeNumber) == NULL)
                    {
                        if (a->getType() == "tangent")
                        {
                            geo->setVertexAttribArray(attributeNumber, tsg->getTangentArray());
                            geo->setVertexAttribBinding(attributeNumber, osg::Geometry::BIND_PER_VERTEX);
                        }
                        if (a->getType() == "binormal")
                        {
                            geo->setVertexAttribArray(attributeNumber, tsg->getBinormalArray());
                            geo->setVertexAttribBinding(attributeNumber, osg::Geometry::BIND_PER_VERTEX);
                        }
                        if (a->getType() == "normal")
                        {
                            geo->setVertexAttribArray(attributeNumber, tsg->getNormalArray());
                            geo->setVertexAttribBinding(attributeNumber, osg::Geometry::BIND_PER_VERTEX);
                        }
                    }
                }
            }
        }
    }
    else
    {
        osg::StateSet *stateset = geode->getOrCreateStateSet();

        if (transparent)
        {
            osg::StateSet *drawableSS = drawable->getOrCreateStateSet();
            drawableSS->setMode(GL_BLEND, osg::StateAttribute::ON);
            drawableSS->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
            // Disable writing to depth buffer.
            osg::Depth *depth = new osg::Depth;
            depth->setWriteMask(false);
            drawable->getOrCreateStateSet()->setAttributeAndModes(depth, osg::StateAttribute::ON);
        }
        if (opaque)
        {
            osg::StateSet *drawableSS = drawable->getOrCreateStateSet();
            drawableSS->setMode(GL_BLEND, osg::StateAttribute::OFF);
            drawableSS->setRenderingHint(osg::StateSet::OPAQUE_BIN);
            // Enable writing to depth buffer.
            osg::Depth *depth = new osg::Depth;
            depth->setWriteMask(true);
            drawable->getOrCreateStateSet()->setAttributeAndModes(depth, osg::StateAttribute::ON);
            stateset->setNestRenderBins(false);
        }
        if (cullFace != -1)
        {
            if (cullFace != 0)
            {
                osg::CullFace *cF = new osg::CullFace();
                cF->setMode((osg::CullFace::Mode)cullFace);
                stateset->setAttributeAndModes(cF, osg::StateAttribute::ON);
            }
            else
            {
                osg::CullFace *cF = new osg::CullFace();
                stateset->setAttributeAndModes(cF, osg::StateAttribute::OFF);
            }
        }
        if (program.get() == NULL)
        {
            program = new osg::Program;
            if (fragmentShader.get())
                program->addShader(fragmentShader.get());
            if (geometryShader.get())
            {
                program->addShader(geometryShader.get());
                program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, geomParams[0]);
                program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, geomParams[1]);
                program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, geomParams[2]);
            }
            if (vertexShader.get())
                program->addShader(vertexShader.get());
            if (tessControlShader.get())
                program->addShader(tessControlShader.get());
            if (tessEvalShader.get())
                program->addShader(tessEvalShader.get());

            std::list<coVRAttribute *>::iterator it;
            for (it = attributes.begin(); it != attributes.end(); it++)
            {
                coVRAttribute *a = *it;
                int attributeNumber = atoi(a->getValue().c_str());
                program->addBindAttribLocation(a->getName(), attributeNumber);
            }
        }
        stateset->setAttributeAndModes(program.get(), osg::StateAttribute::ON);

        std::list<coVRUniform *>::iterator it;
        for (it = uniforms.begin(); it != uniforms.end(); it++)
        {
            if ((*it)->isUnique())
            {
                if (instance == NULL)
                    instance = new coVRShaderInstance(drawable);
                instance->addUniform(*((*it)->uniform.get()));
            }
            else
            {
                stateset->addUniform((*it)->uniform.get());
            }
            if ((*it)->texture.get()) // if we have a texture
            {
                int stage = 0;
                // set texture only, if not already available or if overwrite is set
                sscanf((*it)->getValue().c_str(), "%d", &stage);
                if (stateset->getTextureAttribute(stage, osg::StateAttribute::TEXTURE) == NULL || (*it)->doOverwrite())
                {
                    stateset->setTextureAttributeAndModes(stage, (*it)->texture.get(), osg::StateAttribute::ON);
                }
            }
        }
        if (instance != NULL)
        {
            std::list<osg::ref_ptr<osg::Uniform> >::iterator it;
            for (it = instance->getUniforms().begin(); it != instance->getUniforms().end(); it++)
            {
                stateset->addUniform((*it).get());
            }
            instances.push_back(instance);
        }
    }
    return instance;
}

void coVRShader::setNumVertices(int nv)
{
    if (getProgram().valid())
    {
        getProgram()->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, nv);
        if (getGeometryShader())
            getGeometryShader()->dirtyShader();
    }
    geomParams[0] = nv;
}
void coVRShader::setInputType(int t)
{
    if (getProgram().valid())
    {
        getProgram()->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, t);
        if (getGeometryShader())
            getGeometryShader()->dirtyShader();
    }
    geomParams[1] = t;
}
void coVRShader::setOutputType(int t)
{
    if (getProgram().valid())
    {
        getProgram()->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, t);
        if (getGeometryShader())
        {
            getGeometryShader()->dirtyShader();
        }
    }
    geomParams[2] = t;
}

//coVRShader::coVRShader(TokenBuffer &tb)
//{
//   tb << name;
//   coVRShaderList::instance()->push_back(this);
//}

coVRShader::~coVRShader()
{
}

coVRShaderList::coVRShaderList()
{
    projectionMatrix = new osg::Uniform("Projection", osg::Matrix::translate(100, 0, 0));
    lightMatrix = new osg::Uniform("Light", osg::Matrix::translate(100, 0, 0));
    if (cover)
    {
        timeUniform = new osg::Uniform("Time", (int)(cover->frameTime() * 1000.0));
        durationUniform = new osg::Uniform("Duration", (int)(cover->frameDuration() * 1000.0));
        viewportWidthUniform = new osg::Uniform("ViewportWidth", cover->frontWindowHorizontalSize);
        viewportHeightUniform = new osg::Uniform("ViewportHeight", cover->frontWindowVerticalSize);
    }
    else
    {
        timeUniform = new osg::Uniform("Time", 1000);
        durationUniform = new osg::Uniform("Duration", 1);
        viewportWidthUniform = new osg::Uniform("ViewportWidth", 1024);
        viewportHeightUniform = new osg::Uniform("ViewportHeight", 768);
    }
    stereoUniform = new osg::Uniform("Stereo", 0);
}

void coVRShaderList::loadMaterials()
{
    const char *coviseDir = getenv("COVISEDIR");
    if (coviseDir != NULL)
    {
        std::string buf = coviseDir;
#ifdef WIN32
        buf += "\\share\\covise\\materials";
#else
        buf += "/share/covise/materials";
#endif
        coDirectory *dir = coDirectory::open(buf.c_str());
        for (int i = 0; dir && i < dir->count(); i++)
        {
            if (dir->match(dir->name(i), "*.xml"))
            {
                new coVRShader(dir->name(i), "");
            }
        }
    }
}
void coVRShaderList::init()
{
    osg::Geode *geodeShaderL = new osg::Geode;
    geodeShaderL->setNodeMask(Isect::Left);
    ShaderNode *shaderL;
    shaderL = new ShaderNode(ShaderNode::Left);
    shaderL->setUseDisplayList(false);
    osg::StateSet *statesetBackgroundBin = new osg::StateSet();
    statesetBackgroundBin->setRenderBinDetails(-2, "RenderBin");
    statesetBackgroundBin->setNestRenderBins(false);
    shaderL->setStateSet(statesetBackgroundBin);
    geodeShaderL->addDrawable(shaderL);
    cover->getScene()->addChild(geodeShaderL);

    osg::Geode *geodeShaderR = new osg::Geode;
    geodeShaderR->setNodeMask(Isect::Right);
    ShaderNode *shaderR;
    shaderR = new ShaderNode(ShaderNode::Right);
    shaderR->setUseDisplayList(false);
    shaderR->setStateSet(statesetBackgroundBin);
    geodeShaderR->addDrawable(shaderR);
    cover->getScene()->addChild(geodeShaderR);
}

coVRShader *coVRShaderList::add(const std::string &name, std::string &dirName)
{
    return new coVRShader(name, dirName);
}

coVRShader *coVRShaderList::get(const std::string &n, std::map<std::string, std::string> *params)
{
    std::string basename = n;
    if (basename.rfind(".xml") == (basename.length() - 4))
        basename = std::string(basename, 0, basename.length() - 4);
    coVRShaderList::iterator it;
    for (it = begin(); it != end(); it++)
    {
        if ((*(it))->getName() == basename)
        {
            std::list<coVRUniform *> unilist = (*it)->getUniforms();
            std::map<std::string, std::string>::iterator itparam;
            if (params != NULL)
                for (itparam = params->begin(); itparam != params->end(); itparam++)
                {
                    osg::Uniform *paramUniform = (*it)->getUniform((*itparam).first);
                    if (paramUniform)
                    {
                        std::list<coVRUniform *>::iterator itcoUniform;
                        for (itcoUniform = unilist.begin(); itcoUniform != unilist.end(); itcoUniform++)
                        {
                            if ((*itcoUniform)->uniform == paramUniform)
                            {
                                (*itcoUniform)->setValue((*itparam).second.c_str());
                                break;
                            }
                        }
                    }
                }
            return *it;
        }
    }
    return NULL;
}

coVRShaderList *coVRShaderList::instance()
{
    static coVRShaderList *singleton = NULL;
    if (!singleton)
    {
        singleton = new coVRShaderList;
        singleton->loadMaterials();
    }
    return singleton;
}

void coVRShaderList::setData(TokenBuffer &tb)
{
    std::string name;
    tb >> name;
    coVRShader *shader = get(name);
    if (shader)
    {
        shader->setData(tb);
    }
}
osg::Uniform *coVRShaderList::getTime()
{
    return timeUniform.get();
}
//ab hier neu
osg::Uniform *coVRShaderList::getLight()
{
    return lightMatrix.get();
}
osg::Uniform *coVRShaderList::getProjection()
{
    return projectionMatrix.get();
}
//bis hier neu
osg::Uniform *coVRShaderList::getDuration()
{
    return durationUniform.get();
}

osg::Uniform *coVRShaderList::getViewportWidth()
{
    char str[200];
    sprintf(str, "COVER.WindowConfig.Window:%d", 0);

    viewportWidthUniform->set(coCoviseConfig::getInt("width", str, 1024));
    return viewportWidthUniform.get();
}

osg::Uniform *coVRShaderList::getViewportHeight()
{
    char str[200];
    sprintf(str, "COVER.WindowConfig.Window:%d", 0);

    viewportHeightUniform->set(coCoviseConfig::getInt("height", str, 768));
    return viewportHeightUniform.get();
}

osg::Uniform *coVRShaderList::getStereo()
{
    return stereoUniform.get();
}

void coVRShaderList::update()
{
    static double firstFrameTime = 0.0;
    if (firstFrameTime == 0.0)
        firstFrameTime = cover->frameTime();
    timeUniform->set((int)((cover->frameTime() - firstFrameTime) * 1000.0));
    durationUniform->set((int)(cover->frameDuration() * 1000.0));

    //neu--------------------------------------------------------------------------------------------------------------------
    //Lichtposition und Projektionsmatrix berechnen

    //�berpr�fung ob getScaleTransform bereits gesetzt wurde (siehe VRSceneGraph) und Aktuelle BaseMat berechnen/holen (Ber.wie in pluginsupport)
    //BaseMat transformiert Objekt- in Welt-Koordinaten
    osg::Matrix BMat;
    osg::Matrix BMatLig;
    if (VRSceneGraph::instance()->getScaleTransform() != NULL)
    {
        BMat = VRSceneGraph::instance()->getScaleTransform()->getMatrix();
        osg::Matrix transformMatrix = VRSceneGraph::instance()->getTransform()->getMatrix();
        BMat.postMult(transformMatrix);
        BMatLig = VRSceneGraph::instance()->getScaleTransform()->getMatrix();
    }
    //Viewermat und Inverse berechnen: Base*Viewer entspricht um viewer-position verschobene Weltkoordinaten (bzw. entspricht modelviewmat)
    osg::Matrix ViewMat = cover->getViewerMat();
    osg::Matrix InvViewMat;
    InvViewMat.invert(ViewMat);

    //Rotation um x-Achse um 90� mit einbeziehen
    osg::Matrix Rot;
    Rot.makeRotate(osg::DegreesToRadians(-90.0), 1, 0, 0);

    osg::Matrix WorldViewMat = InvViewMat * Rot;

    //Nur Rotation in RotOnly speichern und Inverse bilden
    osg::Matrix RotOnly = WorldViewMat;
    RotOnly(3, 0) = 0;
    RotOnly(3, 1) = 0;
    RotOnly(3, 2) = 0;
    RotOnly(3, 3) = 1;

    osg::Matrix InvRot;
    InvRot.invert(RotOnly);

    //neue Weltkoordinaten berechnen und diese der Light-Uniform zuweisen auf die in Shader zugegriffen wird (Lichtposition)
    osg::Matrix nWorldViewMat;
    nWorldViewMat = (BMat * WorldViewMat * InvRot) * cover->invEnvCorrectMat;
    osg::Matrix InvnWorldViewMat;
    InvnWorldViewMat.invert(nWorldViewMat);

    osg::Matrix nWVMRotOnly = nWorldViewMat;
    nWVMRotOnly(3, 0) = 0;
    nWVMRotOnly(3, 1) = 0;
    nWVMRotOnly(3, 2) = 0;
    nWVMRotOnly(3, 3) = 1;
    /*
   osg::Matrix Translig;
   Translig.makeTranslate(0,200,0);
   osg::Matrix Rotlig;
   Rotlig.makeRotate(osg::DegreesToRadians(90.0),1,0,0);
   lightMatrix->set(nWVMRotOnly*Translig);*/

    //Projektionsmatrix berechnen, mit entsprechender Translation und Rotation
    osg::Matrix ProjMat;
    ProjMat.makePerspective(160.0, (1024 / 768), 1.0, 10000); //(fovy, aspectratio, znear, zfar), Frustum ist in neg-z-Richtung ausgerichtet
    osg::Matrix Scalepro = Scalepro.scale(5, 5, 5); //Skalierung*Rotation*Translation
    osg::Matrix Rotpro;
    Rotpro.makeRotate(osg::DegreesToRadians(180.0), 0, 0, 1); //Rotation um 180� um z-Achse um Pfeiltextur richtig auszurichten
    osg::Matrix Transpro = Transpro.translate(0, -4000, 500); //alt pro: 0,0,100; neu - wegen translation in vrml-file muss hier auch eine durchgef�hrt werden (frustum muss von oben auf fahrbahn/wand schauen)
    projectionMatrix->set(InvnWorldViewMat * Scalepro * Transpro * ProjMat);

    osg::Matrix Translig = Translig.translate(0, 0, 300); //alt lig: 0,400,300 - neu ohne y weil versch der fahrbahn etc
    lightMatrix->set((BMat * WorldViewMat * InvRot) * cover->invEnvCorrectMat * Translig); //korrekt bis auf skalierung - mit ProjMat oder InvnWorldViewMat mult bringt nur verschiebung mit kamera

    /*osg::Matrix ProjMat;
   ProjMat.makePerspective(160.0,(1024/768),1.0,10000.0);
    //Drehung um -90� um x-Achse damit Frustum in y-Richtung zeigt
   
   osg::Matrix nWorldViewMat;
   nWorldViewMat = BMat * ((WorldViewMat * InvRot) * cover->invEnvCorrectMat);
    
   osg::Matrix InvnWorldViewMat;
   InvnWorldViewMat.invert(nWorldViewMat);  

   projectionMatrix->set(InvnWorldViewMat * ProjMat * Rot);
   lightMatrix->set(nWorldViewMat);*/
    //-------------------------------------------------------------------------------------------
    /*Erzeugen einer neuen mat Matrix und initialisieren der light- und der projection matrix durch zuweisen der BaseMat.
   Diese Matrizen liegen momentan im Ursprung.
   Spaeter sollen die Positionen sich mit dem Fahrzeug mitbewegen - also andere Werte wie BaseMat zuweisen
        
   //osg::Matrix BMat;
   //BMat = cover->getBaseMat();
   //osg::Matrix InvBMat;
   //InvBMat.invert(BMat);

   osg::Matrixd projmat;
   projmat.makePerspective(160.0,2.0,1.0,500.0);
   
   osg::Matrixd projmat(2,0,0,0,
						-3,-3,3,1,
						0,2,0,0,
						0,0,-4,0);  frustum in y-richtung mit n=1, f=2
	projectionMatrix->set(projmat);
						mat(0.2,0,0,0,
							-3,-3,1.00002,1,
							0,0.2,0,0,
							0,0, -0.200002,0); frustum in y-richtung mit n=0.1, f=10000
   osg::Matrixd mat;
   mat = cover->getBaseMat();
   lightMatrix->set(mat);
   
   osg::Matrixd mat;
   lightMatrix->set(osg::Matrix::translate(0,0,100));

   	cout<<"lighmat="<<ligmat<<std::endl;
	cout<<"------"<<std::endl;
	cout<<"projemat="<<projmat<<std::endl;
	cout<<"------"<<std::endl;
	lightMatrix,16;
	*coVRSceneView->*cover->invEnvCorrectMat
	  osg::Matrix ligposmat(1,0,0,0,
						0,1,0,0,
						0,0,1,0,
						0,0,-2000,1);

   osg::Matrix Rotonly(1,0,0,0,
						0,0,-1,0,
						0,1,0,0,
						0,0,0,1); 
   osg::Matrix invRot(1,0,0,0,
						0,0,1,0,
						0,-1,0,0,
						0,0,0,1);
    osg::Matrix Transonly(1,0,0,0,
						0,1,0,0,
						0,0,1,0,
						0,0,-2000,1);
	
	
   osg::Matrix BMat = cover ->getBaseMat();
  
   osg::Matrix ViewMat= cover->getViewerMat();
   
   osg::Matrix WorldViewMat = BMat*ViewMat;   
     
   //zweiter Schritt: verschobene neue weltkoordinaten berechnen
   osg::Matrix RotOnly = WorldViewMat;
   RotOnly(3,0)=0;
   RotOnly(3,1)=0;
   RotOnly(3,2)=0;
   RotOnly(3,3)=1;
   
   osg::Matrix InvRot;
   InvRot.invert(RotOnly);

   osg::Matrix nWorldViewMat;
   nWorldViewMat = ((WorldViewMat*InvRot)*cover->invEnvCorrectMat);
      lightMatrix->set(nWorldViewMat);
   bis hier alt

   osg::Matrix TransOnly = WorldViewMat;
   TransOnly(0,0)=1;
   TransOnly(0,1)=0;
   TransOnly(0,2)=0;
   TransOnly(0,3)=0;

   TransOnly(1,0)=0;
   TransOnly(1,1)=1;
   TransOnly(1,2)=0;
   TransOnly(1,3)=0;

   TransOnly(2,0)=0;
   TransOnly(2,1)=0;
   TransOnly(2,2)=1;
   TransOnly(2,3)=0;
   
   osg::Matrix npm;
   
   Moeglicherweise dies als ALternative, da in viewermat nur translation und nicht rotation beruecksichtigt wird...test entspricht hier der modelview
   osg::Matrix test(1,0,0,0,
						0,0,1,0,
						0,-1,0,0,
						0,2000,0,1);
    lightMatrix->set(test*InvRot*cover->invEnvCorrectMat);
   
   
   
   npm=cover->envCorrectMat *rotonly * *(proj.get());
   
   neu---------------------------------------------------------------------------------------*/

    if (cover->frontWindowHorizontalSize > 0)
    {
        viewportWidthUniform->set(cover->frontWindowHorizontalSize);
        viewportHeightUniform->set(cover->frontWindowVerticalSize);
    }
}

coVRShaderInstance::coVRShaderInstance(osg::Drawable *d)
{
    myDrawable = d;
}
coVRShaderInstance::~coVRShaderInstance()
{
}
void coVRShaderInstance::addUniform(const osg::Uniform &u)
{
    uniforms.push_back(new osg::Uniform(u));
}
osg::Uniform *coVRShaderInstance::getUniform(const std::string &name)
{
    std::list<osg::ref_ptr<osg::Uniform> >::iterator it;
    for (it = uniforms.begin(); it != uniforms.end(); it++)
    {
        if (it->get()->getName() == name)
            return it->get();
    }
    return NULL;
}

ShaderNode::ShaderNode(StereoView v)
{
    theNode = this;
    view = v;
}

ShaderNode::~ShaderNode()
{
    theNode = NULL;
}

ShaderNode *ShaderNode::theNode = NULL;

/** Clone the type of an object, with Object* return type.
Must be defined by derived classes.*/
osg::Object *ShaderNode::cloneType() const
{
    return new ShaderNode(view);
}

/** Clone the an object, with Object* return type.
Must be defined by derived classes.*/
osg::Object *ShaderNode::clone(const osg::CopyOp &) const
{
    return new ShaderNode(view);
}

void ShaderNode::drawImplementation(osg::RenderInfo &renderInfo) const
{
    const unsigned int contextID = renderInfo.getState()->getContextID();
    const osg::GL2Extensions *extensions = osg::GL2Extensions::Get(contextID, true);
    if (!extensions->isGlslSupported())
        return;

    if (view == ShaderNode::Left)
        coVRShaderList::instance()->getStereo()->set(0);
    else
        coVRShaderList::instance()->getStereo()->set(1);
}
