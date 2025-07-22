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
#include "coVRAnimationManager.h"
#include "coVRLighting.h"
#include "coVRConfig.h"

#include <osg/Uniform>
#include <osg/Program>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Image>
#include <osg/Texture1D>
#include <osg/Texture2D>
#include <osg/Texture3D>
#include <osg/TextureRectangle>
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

static std::array<int, 13> glslVersions{110, 120, 130, 140, 150, 330, 400, 410, 420, 430, 440, 450, 460};

coVRUniform::coVRUniform(const coVRShader *s, const std::string &n, const std::string &t, const std::string &v)
{
    shader = s;
    name = n;
    type = t;
    value = v;
    overwrite = false;
    unique = false;
    uniform = coVRShaderList::instance()->getGlobalUniform(name);
    if (uniform == nullptr)
    {
        if (type == "bool")
        {
            bool b = true;
            if (value == "false")
                b = false;
            char* end = nullptr;
            if (strtod(value.c_str(), &end) == 0 && end != value.c_str())
                b = false;
            uniform = new osg::Uniform(name.c_str(), b);
        }
        else if (type == "float")
        {
            float f = (float)strtod(value.c_str(), NULL);
            uniform = new osg::Uniform(name.c_str(), f);
        }
        else if (type == "int")
        {
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
        else if (type == "sampler1D" || type == "sampler2D" || type == "sampler3D" || type == "samplerCube" || type == "sampler2DRect")
        {
            int texUnit = atoi(value.c_str());
            uniform = new osg::Uniform(name.c_str(), texUnit);
        }
        else if (type == "dmat4")
        {
            osg::Matrixd m;
            double values[16];
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
                sscanf(value.c_str(), "%lf %lf %lf %lf  %lf %lf %lf %lf  %lf %lf %lf %lf  %lf %lf %lf %lf", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8], &values[9], &values[10], &values[11], &values[12], &values[13], &values[14], &values[15]);
                uniform->set(osg::Matrix(values));
            }
        }
        else if (type == "mat4")
        {
            osg::Matrixf m;

            if (name == "Light")
            {
                uniform = coVRShaderList::instance()->getLight();
            }
            else
            {
                uniform = new osg::Uniform(name.c_str(), osg::Matrixf::identity());
            }
            if (name == "Projection")
            {
                uniform = coVRShaderList::instance()->getProjection();
            }
            if (strcasecmp(value.c_str(), "identity") == 0)
            {
                uniform->set(osg::Matrixf::identity());
            }
            else
            {
                float values[16];
                sscanf(value.c_str(), "%f %f %f %f  %f %f %f %f  %f %f %f %f  %f %f %f %f", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8], &values[9], &values[10], &values[11], &values[12], &values[13], &values[14], &values[15]);
                uniform->set(osg::Matrixf(values));
            }
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
                if (type == "sampler2DRect")
                {
                    texture = new osg::TextureRectangle;

                    texture->setWrap(osg::Texture::WRAP_R, getWrapMode());
                    texture->setWrap(osg::Texture::WRAP_S, getWrapMode());
                    texture->setWrap(osg::Texture::WRAP_T, getWrapMode());
                }
            }
            texture->setImage(i, image);
            if (type == "samplerCube")
                cubeMapFiles[i] = fn;
        }
    }
}

void coVRUniform::setValue(osg::Matrixd m)
{
    char ms[1600];
    sprintf(ms, "%lf %lf %lf %lf  %lf %lf %lf %lf  %lf %lf %lf %lf  %lf %lf %lf %lf", m(0, 0), m(0, 1), m(0, 2), m(0, 3), m(1, 0), m(1, 1), m(1, 2), m(1, 3), m(2, 0), m(2, 1), m(2, 2), m(2, 3), m(3, 0), m(3, 1), m(3, 2), m(3, 3));
    value = ms;
    uniform->set(m);
}
void coVRUniform::setValue(osg::Matrixf m)
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

void coVRUniform::setValue(bool b)
{
    char fs[100];
    sprintf(fs, "%s", b ? "true" : "false");
    value = fs;
    uniform->set(b);
}

void coVRUniform::setValue(int i)
{
    char fs[100];
    sprintf(fs, "%d", i);
    value = fs;
    uniform->set(i);
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
    if (type == "bool")
    {
        bool b = !(strcmp(val,"false")==0 || strtod(val, NULL)==0);
        uniform = new osg::Uniform(name.c_str(), b);
    }
    else if (type == "float")
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
    else if (type == "dmat4")
    {
        double values[16];
        if (strcasecmp(val, "identity") == 0)
        {
            uniform = new osg::Uniform(name.c_str(), osg::Matrixd::identity());
        }
        else
        {
            sscanf(val, "%lf %lf %lf %lf  %lf %lf %lf %lf  %lf %lf %lf %lf  %lf %lf %lf %lf", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8], &values[9], &values[10], &values[11], &values[12], &values[13], &values[14], &values[15]);
            uniform = new osg::Uniform(name.c_str(), osg::Matrixd(values));
        }
    }
    else if (type == "mat4")
    {
        float values[16];
        if (strcasecmp(val, "identity") == 0)
        {
            uniform = new osg::Uniform(name.c_str(), osg::Matrixf::identity());
        }
        else
        {
            sscanf(val, "%f %f %f %f  %f %f %f %f  %f %f %f %f  %f %f %f %f", &values[0], &values[1], &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8], &values[9], &values[10], &values[11], &values[12], &values[13], &values[14], &values[15]);
            uniform = new osg::Uniform(name.c_str(), osg::Matrixf(values));
        }
    }
    else if (type == "sampler2D" || type == "sampler1D" || type == "sampler3D" || type == "samplerCube" || type == "sampler2DRect")
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

coVRShader::coVRShader(const std::string &n, const std::string &d, const std::string &defines): defines(defines)
{
    wasCloned = false;
    name = n;
    dir = d;
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

coVRShader::coVRShader(const coVRShader &other)
: name(other.name)
, fileName(other.fileName)
, dir(other.dir)
, defines(other.defines)
, wasCloned(true)
, geometryShader(other.geometryShader)
, transparent(other.transparent)
, opaque(other.opaque)
, cullFace(other.cullFace)
{
    for (int i=0; i<3; ++i)
        geomParams[i] = other.geomParams[i];
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

class XmlAttribute
{
public:
    XmlAttribute(const std::string& attribute, xercesc::DOMElement *node)
    {
        XMLCh *t1 = nullptr;
        auto s =
            xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode(attribute.c_str())));
        if (s)
        {
            m_value = s;
        }
        xercesc::XMLString::release(&t1);
        xercesc::XMLString::release(&s);
    }
    ~XmlAttribute() {}
    bool operator==(const std::string &other) const { return m_value == other; }
    operator bool() const { return !m_value.empty(); }
    operator std::string() const { return m_value; }
    const char *c_str() const { return m_value.c_str(); }

private:
    std::string m_value;
};

std::ostream &operator<<(std::ostream &os, const XmlAttribute &attr)
{
    os << attr.c_str();
    return os;
}

std::string prependPreamble(std::string code, const std::string &preamble)
{
    // prepend defines and fragment shader library, but keep #version on first line
    size_t start = code.find_first_not_of(" \t\n\r");
    code = code.substr(start);

    std::string::size_type lineend = code.find_first_of("\n\r");
    if (lineend != std::string::npos)
    {
        // retain space/newline after #version
        ++lineend;
    }
    std::string version = code.substr(0, lineend);
    if (version.find("#version") == std::string::npos)
    {
        version.clear();
    }
    else
    {
        code = code.substr(lineend);
    }

    return version + preamble + code;
}

struct VersionInfo
{
    int min = -1;
    int max = -1;
    std::string profile;
};

VersionInfo parseVersion(xercesc::DOMElement *node, const std::string &name)
{
    VersionInfo info;
    std::string code = "";
    XmlAttribute value("value", node);
    XmlAttribute min("min", node);
    XmlAttribute max("max", node);
    XmlAttribute profile("profile", node);
    if (value)
    {
        info.min = info.max = atoi(value.c_str());
        if (min || max)
        {
            std::cerr << "WARNING: ignoring min/max attributes for version as value is set in shader " << name
                      << std::endl;
        }
    }
    else
    {
        if (min)
            info.min = atoi(min.c_str());
        if (max)
            info.max = atoi(max.c_str());
    }
    if (profile)
    {
        info.profile = profile.c_str();
    }
    return info;
}

std::string parseProgram(xercesc::DOMElement *node, const std::function<std::string(const std::string &)> &findAsset,
                         const std::string &name, const std::string &programType)
{
    std::string code = "";
    XmlAttribute value("value", node);
    if (value)
    {
        std::string filename = findAsset(value);
        if (!filename.empty())
        {
            std::ifstream t(filename.c_str());
            std::stringstream buffer;
            buffer << t.rdbuf();
            code = buffer.str();
            if (code.empty())
                cerr << "WARNING: empty " << programType << " program in " << filename << " for shader " << name
                     << std::endl;
        }
        else
        {
            cerr << "WARNING: could not find " << programType << " program " << filename << " for shader " << name
                 << std::endl;
        }
    }
    if (code == "")
    {
        char *c = xercesc::XMLString::transcode(node->getTextContent());
        if (c && c[0] != '\0')
            code = c;

        xercesc::XMLString::release(&c);
        if (code.empty())
            cerr << "WARNING: empty " << programType << " program for shader " << name << std::endl;
    }
    return code;
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
        rootElement = xmlDoc->getDocumentElement();
    
    if (rootElement)
    {
        XmlAttribute transp("transparent", rootElement);
        transparent = transp == "true";
        XmlAttribute op("opaque", rootElement);
        opaque = op == "true";
        XmlAttribute cullString("cullFace", rootElement);

        if (cullString == "true" || cullString == "on")
            cullFace = osg::CullFace::BACK;
        else if (cullString == "back")
            cullFace = osg::CullFace::BACK;
        else if (cullString == "front")
            cullFace = osg::CullFace::FRONT;
        else if (cullString == "front_and_back")
            cullFace = osg::CullFace::FRONT_AND_BACK;
        else if (cullString == "none" || cullString == "off" || cullString == "false")
            cullFace = 0;
        else if (cullString)
            cerr << "invalid cullFace value \"" << cullString << "\"" << std::endl;

        xercesc::DOMNodeList *nodeList = rootElement->getChildNodes();
        std::string preamble;
        for (size_t i = 0; i < nodeList->getLength(); ++i)
        {
            xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
            if (!node)
                continue;
            char *tagName = xercesc::XMLString::transcode(node->getTagName());
            if (tagName)
            {
                if (strcmp(tagName, "preamble") == 0)
                {
                    preamble = parseProgram(node, std::bind(&coVRShader::findAsset, this, std::placeholders::_1), name,
                                            "preamble");
                }
                if (strcmp(tagName, "version") == 0)
                {
                    auto info = parseVersion(node, name);
                    if (info.min > 0)
                    {
                        versionMin = info.min;
                        if (versionMax < versionMin)
                            versionMax = versionMin;
                    }
                    if (info.max > 0)
                    {
                        versionMax = info.max;
                        if (versionMin > versionMax)
                            versionMin = versionMax;
                    }
                    profile = info.profile;
                }
            }
            xercesc::XMLString::release(&tagName);
        }

        if (versionMin > 0 || versionMax > 0)
        {
            if (versionMin < 0)
                versionMin = 110;
            if (versionMax < 0)
                versionMax = 460;

            std::stringstream ss;
            auto versionRange = coVRShaderList::instance()->glslVersion();
            int ver = 110;
            if (versionMin > versionRange.second)
            {
                ver = versionMin;
                std::cerr << "Shader " << name << " not supported by OpenGL version " << versionRange.first << " to "
                          << versionRange.second << std::endl;
            }
            else if (versionMax < versionRange.first)
            {
                ver = versionMax;
                std::cerr << "Shader " << name << " not supported by OpenGL version " << versionRange.first << " to "
                          << versionRange.second << std::endl;
            }
            else
            {
                for (auto &v: glslVersions)
                {
                    if (v < versionRange.first)
                        continue;
                    if (v < versionMin)
                        continue;
                    if (v > versionRange.second)
                        break;
                    if (v > versionMax)
                        break;
                    ver = v;
                }
            }
            ss << "#version " << ver;
            if (ver >= 330 && !profile.empty())
            {
                ss << " " << profile;
            }
            ss << "\n";
            std::string version = ss.str();
            preamble = version + defines + preamble;
        }
        else
        {
            preamble = defines + preamble;
        }

        for (size_t i = 0; i < nodeList->getLength(); ++i)
        {
            xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
            if (!node)
                continue;
            char *tagName = xercesc::XMLString::transcode(node->getTagName());
            if (tagName)
            {
                if (strcmp(tagName, "preamble") == 0)
                {
                }
                else if (strcmp(tagName, "version") == 0)
                {
                }
                else if (strcmp(tagName, "attribute") == 0)
                {
                    XmlAttribute type("type", node);
                    XmlAttribute value("value", node);
                    XmlAttribute attributeName("name", node);
                    if (type && value && attributeName)
                        attributes.push_back(new coVRAttribute(attributeName.c_str(), type.c_str(), value.c_str()));
                }
                else if (strcmp(tagName, "uniform") == 0)
                {
                    XmlAttribute type("type", node);
                    XmlAttribute value("value", node);
                    XmlAttribute uniformName("name", node);
                    XmlAttribute minValue("min", node);
                    XmlAttribute maxValue("max", node);
                    XmlAttribute textureName("texture", node);
                    XmlAttribute texture1Name("texture1", node);
                    XmlAttribute overwrite("overwrite", node);
                    XmlAttribute unique("unique", node);
                    XmlAttribute wm("wrapMode", node);
                    if (type && value && uniformName)
                    {
                        coVRUniform *u = new coVRUniform(this, uniformName.c_str(), type.c_str(), value.c_str());
                        uniforms.push_back(u);
                        u->setMin(minValue);
                        u->setMax(maxValue);
                        u->setOverwrite(overwrite == "true");
                        u->setUnique(unique == "true");
                        if (wm)
                            u->setWrapMode(wm);
                        if (textureName)
                            u->setTexture(textureName.c_str());
                        if (texture1Name)
                        {
                            for (int i = 0; i < 6; i++)
                            {
                                char attrName[100];
                                sprintf(attrName, "texture%d", i + 1);
                                XmlAttribute textureName(attrName, node);
                                u->setTexture(textureName.c_str(), i);
                            }
                        }
                    }
                }
                else if (strcmp(tagName, "fragmentProgram") == 0)
                {
                    std::string code = parseProgram(
                        node, std::bind(&coVRShader::findAsset, this, std::placeholders::_1), name, "fragment");
                    if (!code.empty())
                    {
                        code = prependPreamble(code, preamble);
                        fragmentShader = new osg::Shader(osg::Shader::FRAGMENT, code);
                        fragmentShader->setName(name);
                    }
                }
                else if (strcmp(tagName, "geometryProgram") == 0)
                {
                    XmlAttribute numVertices("numVertices", node);
                    geomParams[0] = atoi(numVertices.c_str());
                    //FIXME glGetIntegerv requires a valid OpenGL context, otherwise: crash
                    //if (geomParams[0] != 0) glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT,&geomParams[0]);
                    if (geomParams[0] == 0)
                        geomParams[0] = 1024;

                    XmlAttribute inputType("inputType", node);
                    if (inputType == "POINTS")
                        geomParams[1] = GL_POINTS;
                    else if (inputType == "LINES")
                        geomParams[1] = GL_LINES;
                    else if (inputType == "LINES_ADJACENCY_EXT")
                        geomParams[1] = GL_LINES_ADJACENCY_EXT;
                    else if (inputType == "TRIANGLES_ADJACENCY_EXT")
                        geomParams[1] = GL_TRIANGLES_ADJACENCY_EXT;
                    else
                        geomParams[1] = GL_TRIANGLES;

                    XmlAttribute outputType("outputType", node);
                    if (outputType == "POINTS")
                        geomParams[2] = GL_POINTS;
                    else if (outputType == "LINES")
                        geomParams[2] = GL_LINES;
                    else if (outputType == "LINE_STRIP")
                        geomParams[2] = GL_LINE_STRIP;
                    else
                        geomParams[2] = GL_TRIANGLE_STRIP;

                    std::string code = parseProgram(
                        node, std::bind(&coVRShader::findAsset, this, std::placeholders::_1), name, "geometry");
                    if (!code.empty())
                    {
                        code = prependPreamble(code, preamble);
                        geometryShader = new osg::Shader(osg::Shader::GEOMETRY, code);
                        geometryShader->setName(name);
                    }
                }
                else if (strcmp(tagName, "vertexProgram") == 0)
                {
                    std::string code = parseProgram(
                        node, std::bind(&coVRShader::findAsset, this, std::placeholders::_1), name, "vertex");
                    if (!code.empty())
                    {
                        code = prependPreamble(code, preamble);
                        vertexShader = new osg::Shader(osg::Shader::VERTEX, code);
                        vertexShader->setName(name);
                    }
                }
                else if (strcmp(tagName, "tessControlProgram") == 0)
                {
                    std::string code = parseProgram(
                        node, std::bind(&coVRShader::findAsset, this, std::placeholders::_1), name, "tessControl");
                    if (!code.empty())
                    {
                        code = prependPreamble(code, preamble);
                        tessControlShader = new osg::Shader(osg::Shader::TESSCONTROL, code);
                        tessControlShader->setName(name);
                    }
                }
                else if (strcmp(tagName, "tessEvalProgram") == 0)
                {
                    std::string code = parseProgram(
                        node, std::bind(&coVRShader::findAsset, this, std::placeholders::_1), name, "tessEval");
                    if (!code.empty())
                    {
                        code = prependPreamble(code, preamble);
                        tessEvalShader = new osg::Shader(osg::Shader::TESSEVALUATION, code);
                        tessEvalShader->setName(name);
                    }
                }
            }
            else
            {
                std::cerr << "ignoring unknown tag" << tagName << " in " << fileName << std::endl;
            }

            xercesc::XMLString::release(&tagName);
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
    auto u = getcoVRUniform(name);
    if (u)
        return u ? u->uniform.get() : nullptr;
}

coVRUniform *coVRShader::getcoVRUniform(const std::string &name)
{
    auto it = std::find_if(uniforms.begin(), uniforms.end(),
                         [&name](const coVRUniform *u) { return u->getName() == name; });
    if (it != uniforms.end())
    {
        return *it;
    }
    return nullptr;
}


void coVRShader::setMatrixUniform(const std::string &name, osg::Matrixd m)
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

void coVRShader::setMatrixUniform(const std::string &name, osg::Matrixf m)
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
        }
        else
        {
            geometryShader->setShaderSource(code.c_str());
        }
        program->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, geomParams[0]);
        program->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, geomParams[1]);
        program->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, geomParams[2]);
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
	XMLCh *t1=NULL;
	XMLCh *t2=NULL;
    impl = xercesc::DOMImplementationRegistry::getDOMImplementation(t1 = xercesc::XMLString::transcode("Core")); xercesc::XMLString::release(&t1);

    std::string ShaderName = name;

    if (ShaderName[0] >= '0' || ShaderName[0] <= '9')
        ShaderName.insert(0, 1, '_');

    for (size_t i = 0; i < ShaderName.length(); i++)
    {
        if (ShaderName[i] == ' ')
            ShaderName[i] = '_';
    }
    xercesc::DOMDocument *document = NULL;

    try
    {
        document = impl->createDocument(0, xercesc::XMLString::transcode(ShaderName.c_str()), 0);
    }
    catch (xercesc::DOMException &ex)
    {
        char *msg = xercesc::XMLString::transcode(ex.getMessage());
        cerr << "ERROR: " << msg << '\n';
        xercesc::XMLString::release(&msg);
        return;
    }
    xercesc::DOMElement *rootElement = document->getDocumentElement();
	if (transparent)
	{
		rootElement->setAttribute(t1 = xercesc::XMLString::transcode("transparent"), t2 = xercesc::XMLString::transcode("true"));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
	}
    if (opaque)
	{
		rootElement->setAttribute(t1 = xercesc::XMLString::transcode("opaque"), t2 = xercesc::XMLString::transcode("true"));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
	}
    if (cullFace == osg::CullFace::BACK)
	{
		rootElement->setAttribute(t1 = xercesc::XMLString::transcode("cullFace"), t2 = xercesc::XMLString::transcode("back"));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
	}
    if (cullFace == osg::CullFace::FRONT)
	{
		rootElement->setAttribute(t1 = xercesc::XMLString::transcode("cullFace"), t2 = xercesc::XMLString::transcode("front"));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
	}
    if (cullFace == osg::CullFace::FRONT_AND_BACK)
	{
		rootElement->setAttribute(t1 = xercesc::XMLString::transcode("cullFace"), t2 = xercesc::XMLString::transcode("front_and_back"));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
	}
    if (cullFace == 0)
	{
		rootElement->setAttribute(t1 = xercesc::XMLString::transcode("cullFace"), t2 = xercesc::XMLString::transcode("off"));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
	}

    for (std::list<coVRUniform *>::iterator it = uniforms.begin(); it != uniforms.end(); ++it)
    {
        xercesc::DOMElement *uniform = document->createElement(t1 = xercesc::XMLString::transcode("uniform"));
		xercesc::XMLString::release(&t1);
        uniform->setAttribute(t1 = xercesc::XMLString::transcode("name"), t2 = xercesc::XMLString::transcode((*it)->getName().c_str()));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
        uniform->setAttribute(t1 = xercesc::XMLString::transcode("type"), t2 = xercesc::XMLString::transcode((*it)->getType().c_str()));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
        uniform->setAttribute(t1 = xercesc::XMLString::transcode("value"), t2 = xercesc::XMLString::transcode((*it)->getValue().c_str()));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
        const std::string *fn = (*it)->getCubeMapFiles();
        if (fn[0].empty())
        {
            if (!(*it)->getTextureFileName().empty())
			{
                uniform->setAttribute(t1 = xercesc::XMLString::transcode("texture"), t2 = xercesc::XMLString::transcode((*it)->getTextureFileName().c_str()));
				xercesc::XMLString::release(&t1);
				xercesc::XMLString::release(&t2);
			}
        }
        else
        {
            for (int i = 0; i < 6; i++)
            {
                char attrName[100];
                sprintf(attrName, "texture%d", i + 1);
                uniform->setAttribute(t1 = xercesc::XMLString::transcode(attrName), t2 = xercesc::XMLString::transcode(fn[i].c_str()));
				xercesc::XMLString::release(&t1);
				xercesc::XMLString::release(&t2);
            }
        }

        if (!(*it)->getMin().empty())
        {
            uniform->setAttribute(t1 = xercesc::XMLString::transcode("min"), t2 = xercesc::XMLString::transcode((*it)->getMin().c_str()));
			xercesc::XMLString::release(&t1);
			xercesc::XMLString::release(&t2);
            uniform->setAttribute(t1 = xercesc::XMLString::transcode("max"), t2 = xercesc::XMLString::transcode((*it)->getMax().c_str()));
			xercesc::XMLString::release(&t1);
			xercesc::XMLString::release(&t2);
        }
        if ((*it)->isUnique())
        {
            uniform->setAttribute(t1 = xercesc::XMLString::transcode("unique"), t2 = xercesc::XMLString::transcode("true"));
			xercesc::XMLString::release(&t1);
			xercesc::XMLString::release(&t2);
        }
        if ((*it)->doOverwrite())
        {
            uniform->setAttribute(t1 = xercesc::XMLString::transcode("overwrite"), t2 = xercesc::XMLString::transcode("true"));
			xercesc::XMLString::release(&t1);
			xercesc::XMLString::release(&t2);
        }
        rootElement->appendChild(uniform);
    }

    for (std::list<coVRAttribute *>::iterator it = attributes.begin(); it != attributes.end(); ++it)
    {
        xercesc::DOMElement *attrib = document->createElement(t1 = xercesc::XMLString::transcode("attribute"));
		xercesc::XMLString::release(&t1);
        attrib->setAttribute(t1 = xercesc::XMLString::transcode("name"), t2 = xercesc::XMLString::transcode((*it)->getName().c_str()));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
        attrib->setAttribute(t1 = xercesc::XMLString::transcode("type"), t2 = xercesc::XMLString::transcode((*it)->getType().c_str()));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
        attrib->setAttribute(t1 = xercesc::XMLString::transcode("value"), t2 = xercesc::XMLString::transcode((*it)->getValue().c_str()));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
        rootElement->appendChild(attrib);
    }
    if (vertexShader.get() != NULL)
    {
        xercesc::DOMElement *vertexProgram = document->createElement(t1 = xercesc::XMLString::transcode("vertexProgram"));
        vertexProgram->setTextContent(t2 = xercesc::XMLString::transcode(vertexShader->getShaderSource().c_str()));
        rootElement->appendChild(vertexProgram);
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
    }
    if (tessControlShader.get() != NULL)
    {
        xercesc::DOMElement *vertexProgram = document->createElement(t1 = xercesc::XMLString::transcode("tessControlProgram"));
        vertexProgram->setTextContent(t2 = xercesc::XMLString::transcode(tessControlShader->getShaderSource().c_str()));
        rootElement->appendChild(vertexProgram);
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
    }
    if (tessEvalShader.get() != NULL)
    {
        xercesc::DOMElement *vertexProgram = document->createElement(t1 = xercesc::XMLString::transcode("tessEvalProgram"));
        vertexProgram->setTextContent(t2 = xercesc::XMLString::transcode(tessEvalShader->getShaderSource().c_str()));
        rootElement->appendChild(vertexProgram);
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
    }
    if (geometryShader.get() != NULL)
    {
        xercesc::DOMElement *geometryProgram = document->createElement(t1 = xercesc::XMLString::transcode("geometryProgram"));
        char numVertices[100]; // no idea what that ment: GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT
        snprintf(numVertices,100, "%d", geomParams[0]);
		xercesc::XMLString::release(&t1);
        geometryProgram->setAttribute(t1 = xercesc::XMLString::transcode("numVertices"), t2 = xercesc::XMLString::transcode(numVertices));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
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

        geometryProgram->setAttribute(t1 = xercesc::XMLString::transcode("inputType"), t2 = xercesc::XMLString::transcode(inputType.c_str()));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);

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

        geometryProgram->setAttribute(t1 = xercesc::XMLString::transcode("outputType"), t2 = xercesc::XMLString::transcode(outputType.c_str()));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
        geometryProgram->setTextContent(t1 = xercesc::XMLString::transcode(geometryShader->getShaderSource().c_str()));
		xercesc::XMLString::release(&t1);
        rootElement->appendChild(geometryProgram);
    }
    if (fragmentShader.get() != NULL)
    {
        xercesc::DOMElement *fragmentProgram = document->createElement(t1 = xercesc::XMLString::transcode("fragmentProgram"));
        fragmentProgram->setTextContent(t2 = xercesc::XMLString::transcode(fragmentShader->getShaderSource().c_str()));
		xercesc::XMLString::release(&t1);
		xercesc::XMLString::release(&t2);
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
    theOutput->setEncoding(t1 = xercesc::XMLString::transcode("utf8"));

    bool written = writer->writeToURI(rootElement, t2 = xercesc::XMLString::transcode(fileName.c_str()));
    if (!written)
        fprintf(stderr, "Material::save info: Could not open file for writing %s!\n", fileName.c_str());
	xercesc::XMLString::release(&t1);
	xercesc::XMLString::release(&t2);
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
                    stateset->setTextureMode(i, GL_TEXTURE_RECTANGLE, osg::StateAttribute::OFF);
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
        program->setName(name);
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
            program->setName(name);
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
			else
			{
				osg::ref_ptr<coTangentSpaceGenerator> coTsg = new coTangentSpaceGenerator;
				//generate assuming box mapping
				coTsg->generate(geo);
				if (!coTsg->getTangentArray()->empty())
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
								geo->setVertexAttribArray(attributeNumber, coTsg->getTangentArray());
								geo->setVertexAttribBinding(attributeNumber, osg::Geometry::BIND_PER_VERTEX);
							}
							if (a->getType() == "binormal")
							{
								geo->setVertexAttribArray(attributeNumber, coTsg->getBinormalArray());
								geo->setVertexAttribBinding(attributeNumber, osg::Geometry::BIND_PER_VERTEX);
							}
							if (a->getType() == "normal")
							{
								geo->setVertexAttribArray(attributeNumber, coTsg->getNormalArray());
								geo->setVertexAttribBinding(attributeNumber, osg::Geometry::BIND_PER_VERTEX);
							}
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
            program->setName(name);
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
    assert(!s_instance);

    projectionMatrix = new osg::Uniform("Projection", osg::Matrixf::translate(100, 0, 0));
    lightMatrix = new osg::Uniform("Light", osg::Matrixf::translate(100, 0, 0));
    lightEnabled.resize(4);
    if (cover)
    {
        timeUniform = new osg::Uniform("Time", (int)(cover->frameTime() * 1000.0));
        timeStepUniform = new osg::Uniform("TimeStep", coVRAnimationManager::instance()->getAnimationFrame());
        durationUniform = new osg::Uniform("Duration", (int)(cover->frameDuration() * 1000.0));
        viewportWidthUniform = new osg::Uniform("ViewportWidth", cover->frontWindowHorizontalSize);
        viewportHeightUniform = new osg::Uniform("ViewportHeight", cover->frontWindowVerticalSize);
        for (size_t i=0; i<lightEnabled.size(); ++i) {
            lightEnabled[i] = new osg::Uniform(("Light" + std::to_string(i) + "Enabled").c_str(), i==0);
        }
    }
    else
    {
        timeUniform = new osg::Uniform("Time", 1000);
        timeStepUniform = new osg::Uniform("TimeStep", 0);
        durationUniform = new osg::Uniform("Duration", 1);
        viewportWidthUniform = new osg::Uniform("ViewportWidth", 1024);
        viewportHeightUniform = new osg::Uniform("ViewportHeight", 768);
        for (size_t i=0; i<lightEnabled.size(); ++i) {
            lightEnabled[i] = new osg::Uniform(("Light" + std::to_string(i) + "Enabled").c_str(), i==0);
        }
    }
    stereoUniform = new osg::Uniform("Stereo", 0);
}

coVRShaderList::~coVRShaderList()
{
    clear();
    s_instance = NULL;
}

std::pair<int, int> coVRShaderList::glslVersion() const
{
    return glslVersionRange;
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
        std::unique_ptr<coDirectory> dir(coDirectory::open(buf.c_str()));
        for (int i = 0; dir && i < dir->count(); i++)
        {
            if (dir->match(dir->name(i), "*.xml"))
            {
                new coVRShader(dir->name(i), dir->path());
            }
        }
    }
}

void coVRShaderList::init(osg::GLExtensions *glext)
{
    int glslVersion = -1;
    if (glext)
    {
        if (glext->isGlslSupported)
        {
            if (cover->debugLevel(1))
                std::cerr << "GLSL supported: version " << glext->glslLanguageVersion << std::endl;
            glslVersion = static_cast<int>(glext->glslLanguageVersion * 100.0 + 0.5);
        }
        else
        {
            std::cerr << "GLSL supported: NO" << std::endl;
        }
    }
    else
    {
        std::cerr << "coVRShaderList::init: no information on GL extensions available" << std::endl;
    }

    glslVersionRange.first = coCoviseConfig::getInt("versionMin", "COVER.GLSL", 110);
    glslVersionRange.second = coCoviseConfig::getInt("versionMax", "COVER.GLSL", glslVersion);

    loadMaterials();

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

coVRShader *coVRShaderList::add(const std::string &name, const std::string &dirName, const std::string &defines)
{
    return new coVRShader(name, dirName, defines);
}

void coVRShaderList::applyParams(coVRShader *shader, std::map<std::string, std::string> *params)
{
    if (!params)
        return;

    std::list<coVRUniform *> unilist = shader->getUniforms();
    for(const auto &param : *params)
    {
        auto uniform = shader->getcoVRUniform(param.first);
        if(uniform)
        {
            uniform->setValue(param.second.c_str());
        }
    }
}

coVRShader *coVRShaderList::getUnique(const std::string &n, std::map<std::string, std::string> *params,
                                      const std::string &defines)
{
    coVRShader *shader = get(n);
    if (!shader)
    {
        return NULL;
    }
    coVRShader *clone = add(shader->fileName, shader->dir, defines);
    clone->wasCloned = true;
    applyParams(clone, params);
    return clone;
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
            if ((*it)->isClone())
                continue;
            applyParams(*it, params);
            return *it;
        }
    }
    return NULL;
}

coVRShaderList *coVRShaderList::s_instance = NULL;
coVRShaderList *coVRShaderList::instance()
{
    if (!s_instance)
    {
        s_instance = new coVRShaderList;
    }
    return s_instance;
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

osg::Uniform* coVRShaderList::getGlobalUniform(const std::string& name)
{
    if (name == "Light0Enabled")
    {
        return getLightEnabled(0);
    }
    else if (name == "Light1Enabled")
    {
        return getLightEnabled(1);
    }
    else if (name == "Light2Enabled")
    {
        return getLightEnabled(2);
    }
    else if (name == "Light3Enabled")
    {
        return getLightEnabled(3);
    }
    else if (name == "Time")
    {
        return getTime();
    }
    else if (name == "TimeStep")
    {
        return getTimeStep();
    }
    else if (name == "Stereo")
    {
        return getStereo();
    }
    else if (name == "Duration")
    {
        return getDuration();
    }
    else if (name == "ViewportWidth")
    {
        return getViewportWidth();
    }
    else if (name == "ViewportHeight")
    {
        return getViewportHeight();
    }
    else
    {
        auto it = globalUniforms.find(name);
        if (it == globalUniforms.end())
            return nullptr;
        else
            return it->second;
    }
}

void coVRShaderList::addGlobalUniform(const std::string&n, osg::Uniform*u)
{
    globalUniforms[n]=u;
    for (const auto& shader : *this)
    {
        for (const auto& un : shader->uniforms)
        {
            if (un->getName() == n)
            {
                un->uniform = u;
            }
        }
    }
}
void coVRShaderList::removeGlobalUniform(osg::Uniform* un)
{
    for (const auto& u : globalUniforms)
    {
        if (u.second == un)
        {
            globalUniforms.erase(u.first);
            break;
        }
    }
}


osg::Uniform *coVRShaderList::getTime()
{
    return timeUniform.get();
}
osg::Uniform *coVRShaderList::getTimeStep()
{
    return timeStepUniform.get();
}
osg::Uniform *coVRShaderList::getLightEnabled(size_t ln)
{
    if (ln > lightEnabled.size())
        return nullptr;
    return lightEnabled[ln];
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
    timeStepUniform->set(coVRAnimationManager::instance()->getAnimationFrame());
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
    
    if(coVRConfig::instance()->getEnvMapMode() == coVRConfig::NONE)
    {
        nWorldViewMat = (BMat * WorldViewMat);
    }
    else
        nWorldViewMat = (BMat * WorldViewMat * InvRot) * cover->invEnvCorrectMat;
    osg::Matrix InvnWorldViewMat;
    InvnWorldViewMat.invert(nWorldViewMat);

    osg::Matrix nWVMRotOnly = nWorldViewMat;
    nWVMRotOnly(3, 0) = 0;
    nWVMRotOnly(3, 1) = 0;
    nWVMRotOnly(3, 2) = 0;
    nWVMRotOnly(3, 3) = 1;

    //Projektionsmatrix berechnen, mit entsprechender Translation und Rotation
    osg::Matrix ProjMat;
    ProjMat.makePerspective(160.0, (1024 / 768), 1.0, 10000); //(fovy, aspectratio, znear, zfar), Frustum ist in neg-z-Richtung ausgerichtet
    osg::Matrix Scalepro = Scalepro.scale(5, 5, 5); //Skalierung*Rotation*Translation
    osg::Matrix Rotpro;
    Rotpro.makeRotate(osg::DegreesToRadians(180.0), 0, 0, 1); //Rotation um 180� um z-Achse um Pfeiltextur richtig auszurichten
    osg::Matrix Transpro = Transpro.translate(0, -4000, 500); //alt pro: 0,0,100; neu - wegen translation in vrml-file muss hier auch eine durchgef�hrt werden (frustum muss von oben auf fahrbahn/wand schauen)
    projectionMatrix->set(osg::Matrixf(InvnWorldViewMat * Scalepro * Transpro * ProjMat));

    osg::Matrix Translig = Translig.translate(0, 0, 300); //alt lig: 0,400,300 - neu ohne y weil versch der fahrbahn etc
    
    if(coVRConfig::instance()->getEnvMapMode() == coVRConfig::NONE)
    {
        lightMatrix->set(osg::Matrixf((BMat * WorldViewMat)  * Translig)); //korrekt bis auf skalierung - mit ProjMat oder InvnWorldViewMat mult bringt nur verschiebung mit kamera
    }
    else
        lightMatrix->set(osg::Matrixf((BMat * WorldViewMat * InvRot) * cover->invEnvCorrectMat * Translig)); //korrekt bis auf skalierung - mit ProjMat oder InvnWorldViewMat mult bringt nur verschiebung mit kamera

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

   for (auto i=0; i<lightEnabled.size(); ++i)
   {
       lightEnabled[i]->set(coVRLighting::instance()->isLightEnabled(i));
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
    const osg::ref_ptr<osg::GL2Extensions> extensions = osg::GL2Extensions::Get(contextID, true);
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 3)
    if (!extensions->isGlslSupported)
        return;
#else
    if (!extensions->isGlslSupported())
        return;
#endif

    if (view == ShaderNode::Left)
        coVRShaderList::instance()->getStereo()->set(0);
    else
        coVRShaderList::instance()->getStereo()->set(1);
}


coTangentSpaceGenerator::coTangentSpaceGenerator()
	: osg::Referenced(),
	T_(new osg::Vec4Array),
	B_(new osg::Vec4Array),
	N_(new osg::Vec4Array)
{
	T_->setBinding(osg::Array::BIND_PER_VERTEX); T_->setNormalize(false);
	B_->setBinding(osg::Array::BIND_PER_VERTEX); T_->setNormalize(false);
	N_->setBinding(osg::Array::BIND_PER_VERTEX); T_->setNormalize(false);
}

coTangentSpaceGenerator::coTangentSpaceGenerator(const coTangentSpaceGenerator &copy, const osg::CopyOp &copyop)
	: osg::Referenced(copy),
	T_(static_cast<osg::Vec4Array *>(copyop(copy.T_.get()))),
	B_(static_cast<osg::Vec4Array *>(copyop(copy.B_.get()))),
	N_(static_cast<osg::Vec4Array *>(copyop(copy.N_.get())))
{
}

void coTangentSpaceGenerator::generate(osg::Geometry *geo)
{
	const osg::Array *vx = geo->getVertexArray();
	const osg::Array *nx = geo->getNormalArray();

	if (!vx) return;


	unsigned int vertex_count = vx->getNumElements();
	T_->assign(vertex_count, osg::Vec4());
	B_->assign(vertex_count, osg::Vec4());
	N_->assign(vertex_count, osg::Vec4());

	unsigned int i; // VC6 doesn't like for-scoped variables

	for (unsigned int pri = 0; pri<geo->getNumPrimitiveSets(); ++pri) {
		osg::PrimitiveSet *pset = geo->getPrimitiveSet(pri);

		unsigned int N = pset->getNumIndices();

		switch (pset->getMode()) {

		case osg::PrimitiveSet::TRIANGLES:
			for (i = 0; i<N; i += 3) {
				compute(pset, vx, nx, i, i + 1, i + 2);
			}
			break;

		case osg::PrimitiveSet::QUADS:
			for (i = 0; i<N; i += 4) {
				compute(pset, vx, nx, i, i + 1, i + 2);
				compute(pset, vx, nx, i + 2, i + 3, i);
			}
			break;

		case osg::PrimitiveSet::TRIANGLE_STRIP:
			if (pset->getType() == osg::PrimitiveSet::DrawArrayLengthsPrimitiveType) {
				osg::DrawArrayLengths *dal = static_cast<osg::DrawArrayLengths *>(pset);
				unsigned int j = 0;
				for (osg::DrawArrayLengths::const_iterator pi = dal->begin(); pi != dal->end(); ++pi) {
					unsigned int iN = static_cast<unsigned int>(*pi - 2);
					for (i = 0; i<iN; ++i, ++j) {
						if ((i % 2) == 0) {
							compute(pset, vx, nx, j, j + 1, j + 2);
						}
						else {
							compute(pset, vx, nx, j + 1, j, j + 2);
						}
					}
					j += 2;
				}
			}
			else {
				for (i = 0; i<N - 2; ++i) {
					if ((i % 2) == 0) {
						compute(pset, vx, nx, i, i + 1, i + 2);
					}
					else {
						compute(pset, vx, nx, i + 1, i, i + 2);
					}
				}
			}
			break;

		case osg::PrimitiveSet::QUAD_STRIP:
			if (pset->getType() == osg::PrimitiveSet::DrawArrayLengthsPrimitiveType) {
				osg::DrawArrayLengths *dal = static_cast<osg::DrawArrayLengths *>(pset);
				unsigned int j = 0;
				for (osg::DrawArrayLengths::const_iterator pi = dal->begin(); pi != dal->end(); ++pi) {
					unsigned int iN = static_cast<unsigned int>(*pi - 2);
					for (i = 0; i<iN; ++i, ++j) {
						if ((i % 2) == 0) {
							compute(pset, vx, nx, j, j + 2, j + 1);
						}
						else {
							compute(pset, vx, nx, j, j + 1, j + 2);
						}
					}
					j += 2;
				}
			}
			else {
				for (i = 0; i<N - 2; ++i) {
					if ((i % 2) == 0) {
						compute(pset, vx, nx, i, i + 2, i + 1);
					}
					else {
						compute(pset, vx, nx, i, i + 1, i + 2);
					}
				}
			}
			break;

		case osg::PrimitiveSet::TRIANGLE_FAN:
		case osg::PrimitiveSet::POLYGON:
			if (pset->getType() == osg::PrimitiveSet::DrawArrayLengthsPrimitiveType) {
				osg::DrawArrayLengths *dal = static_cast<osg::DrawArrayLengths *>(pset);
				unsigned int j = 0;
				for (osg::DrawArrayLengths::const_iterator pi = dal->begin(); pi != dal->end(); ++pi) {
					unsigned int iN = static_cast<unsigned int>(*pi - 2);
					for (i = 0; i<iN; ++i) {
						compute(pset, vx, nx, 0, j + 1, j + 2);
					}
					j += 2;
				}
			}
			else {
				for (i = 0; i<N - 2; ++i) {
					compute(pset, vx, nx, 0, i + 1, i + 2);
				}
			}
			break;

		case osg::PrimitiveSet::POINTS:
		case osg::PrimitiveSet::LINES:
		case osg::PrimitiveSet::LINE_STRIP:
		case osg::PrimitiveSet::LINE_LOOP:
		case osg::PrimitiveSet::LINES_ADJACENCY:
		case osg::PrimitiveSet::LINE_STRIP_ADJACENCY:
			break;

		default: OSG_WARN << "Warning: coTangentSpaceGenerator: unknown primitive mode " << pset->getMode() << "\n";
		}
	}

	// normalize basis vectors and force the normal vector to match
	// the triangle normal's direction
	unsigned int attrib_count = vx->getNumElements();
	for (i = 0; i<attrib_count; ++i) {
		osg::Vec4 &vT = (*T_)[i];
		osg::Vec4 &vB = (*B_)[i];
		osg::Vec4 &vN = (*N_)[i];

		osg::Vec3 txN = osg::Vec3(vT.x(), vT.y(), vT.z()) ^ osg::Vec3(vB.x(), vB.y(), vB.z());
		bool flipped = txN * osg::Vec3(vN.x(), vN.y(), vN.z()) < 0;

		if (flipped) {
			vN = osg::Vec4(-txN, 0);
		}
		else {
			vN = osg::Vec4(txN, 0);
		}

		vT.normalize();
		vB.normalize();
		vN.normalize();

		vT[3] = flipped ? -1.0f : 1.0f;
	}
	/* TO-DO: if indexed, compress the attributes to have only one
	* version of each (different indices for each one?) */
}

void coTangentSpaceGenerator::compute(osg::PrimitiveSet *pset,
	const osg::Array* vx,
	const osg::Array* nx,
	int iA, int iB, int iC)
{
	iA = pset->index(iA);
	iB = pset->index(iB);
	iC = pset->index(iC);

	osg::Vec3 P1;
	osg::Vec3 P2;
	osg::Vec3 P3;

	int i; // VC6 doesn't like for-scoped variables

	switch (vx->getType())
	{
	case osg::Array::Vec2ArrayType:
		for (i = 0; i < 2; ++i) {
			P1.ptr()[i] = static_cast<const osg::Vec2Array&>(*vx)[iA].ptr()[i];
			P2.ptr()[i] = static_cast<const osg::Vec2Array&>(*vx)[iB].ptr()[i];
			P3.ptr()[i] = static_cast<const osg::Vec2Array&>(*vx)[iC].ptr()[i];
		}
		break;

	case osg::Array::Vec3ArrayType:
		P1 = static_cast<const osg::Vec3Array&>(*vx)[iA];
		P2 = static_cast<const osg::Vec3Array&>(*vx)[iB];
		P3 = static_cast<const osg::Vec3Array&>(*vx)[iC];
		break;

	case osg::Array::Vec4ArrayType:
		for (i = 0; i < 3; ++i) {
			P1.ptr()[i] = static_cast<const osg::Vec4Array&>(*vx)[iA].ptr()[i];
			P2.ptr()[i] = static_cast<const osg::Vec4Array&>(*vx)[iB].ptr()[i];
			P3.ptr()[i] = static_cast<const osg::Vec4Array&>(*vx)[iC].ptr()[i];
		}
		break;

	default:
		OSG_WARN << "Warning: coTangentSpaceGenerator: vertex array must be Vec2Array, Vec3Array or Vec4Array" << std::endl;
	}

	osg::Vec3 N1;
	osg::Vec3 N2;
	osg::Vec3 N3;

	if (nx)
	{
		switch (nx->getType())
		{
		case osg::Array::Vec2ArrayType:
			for (i = 0; i < 2; ++i) {
				N1.ptr()[i] = static_cast<const osg::Vec2Array&>(*nx)[iA].ptr()[i];
				N2.ptr()[i] = static_cast<const osg::Vec2Array&>(*nx)[iB].ptr()[i];
				N3.ptr()[i] = static_cast<const osg::Vec2Array&>(*nx)[iC].ptr()[i];
			}
			break;

		case osg::Array::Vec3ArrayType:
			N1 = static_cast<const osg::Vec3Array&>(*nx)[iA];
			N2 = static_cast<const osg::Vec3Array&>(*nx)[iB];
			N3 = static_cast<const osg::Vec3Array&>(*nx)[iC];
			break;

		case osg::Array::Vec4ArrayType:
			for (i = 0; i < 3; ++i) {
				N1.ptr()[i] = static_cast<const osg::Vec4Array&>(*nx)[iA].ptr()[i];
				N2.ptr()[i] = static_cast<const osg::Vec4Array&>(*nx)[iB].ptr()[i];
				N3.ptr()[i] = static_cast<const osg::Vec4Array&>(*nx)[iC].ptr()[i];
			}
			break;

		default:
			OSG_WARN << "Warning: coTangentSpaceGenerator: normal array must be Vec2Array, Vec3Array or Vec4Array" << std::endl;
		}
	}
	else  // no normal per vertex use the one by face
	{
		N1 = (P2 - P1) ^ (P3 - P1);
		N2 = N1;
		N3 = N1;
	}


	osg::Vec3 V, T1, T2, T3, B1, B2, B3;

	if ((N1.z() > 0.8)|| (N1.z() < -0.8))
	{
		T1 = osg::Vec3(1, 0, 0);
		B1 = osg::Vec3(0, 1, 0);
	}
	else
	{
		if ((N1.y() > 0.8) || (N1.y() < -0.8))
		{
			T1 = osg::Vec3(1, 0, 0);
			B1 = osg::Vec3(0, 0, 1);
		}
		else
		{
			T1 = osg::Vec3(0, 1, 0);
			B1 = osg::Vec3(0, 0, 1);
		}
	}
	if ((N2.z() > 0.8) || (N2.z() < -0.8))
	{
		T2 = osg::Vec3(1, 0, 0);
		B2 = osg::Vec3(0, 1, 0);
	}
	else
	{
		if ((N2.y() > 0.8) || (N2.y() < -0.8))
		{
			T2 = osg::Vec3(1, 0, 0);
			B2 = osg::Vec3(0, 0, 1);
		}
		else
		{
			T2 = osg::Vec3(0, 1, 0);
			B2 = osg::Vec3(0, 0, 1);
		}
	}
	if ((N3.z() > 0.8) || (N3.z() < -0.8))
	{
		T3 = osg::Vec3(1, 0, 0);
		B3 = osg::Vec3(0, 1, 0);
	}
	else
	{
		if ((N3.y() > 0.8) || (N3.y() < -0.8))
		{
			T3 = osg::Vec3(1, 0, 0);
			B3 = osg::Vec3(0, 0, 1);
		}
		else
		{
			T3 = osg::Vec3(0, 1, 0);
			B3 = osg::Vec3(0, 0, 1);
		}
	}


	
	osg::Vec3 tempvec;

	tempvec = N1 ^ T1;
	(*T_)[iA] += osg::Vec4(tempvec ^ N1, 0);

	tempvec = B1 ^ N1;
	(*B_)[iA] += osg::Vec4(N1 ^ tempvec, 0);

	tempvec = N2 ^ T2;
	(*T_)[iB] += osg::Vec4(tempvec ^ N2, 0);

	tempvec = B2 ^ N2;
	(*B_)[iB] += osg::Vec4(N2 ^ tempvec, 0);

	tempvec = N3 ^ T3;
	(*T_)[iC] += osg::Vec4(tempvec ^ N3, 0);

	tempvec = B3 ^ N3;
	(*B_)[iC] += osg::Vec4(N3 ^ tempvec, 0);

	(*N_)[iA] += osg::Vec4(N1, 0);
	(*N_)[iB] += osg::Vec4(N2, 0);
	(*N_)[iC] += osg::Vec4(N3, 0);

}


void opencover::coVRShader::setBoolUniform(const std::string &name, bool b)
{
    std::list<coVRUniform *>::iterator it;
    for (it = uniforms.begin(); it != uniforms.end(); it++)
    {
        if ((*it)->getName() == name)
        {
            (*it)->setValue(b);
        }
    }
}
void opencover::coVRShader::setIntUniform(const std::string &name, int i)
{
    std::list<coVRUniform *>::iterator it;
    for (it = uniforms.begin(); it != uniforms.end(); it++)
    {
        if ((*it)->getName() == name)
        {
            (*it)->setValue(i);
        }
    }
}
