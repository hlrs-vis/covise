/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MyShader.h"

// constructors and destructor
// -----------------------------------------
MyShader::MyShader()
{
    _transparent = false;
}

MyShader::~MyShader()
{
}

// setter and getter
// -----------------------------------------

void MyShader::setFragmentFile(std::string file)
{
    _fragment = file;
}

std::string MyShader::getFragment()
{
    return _fragment;
}

void MyShader::setVertexFile(std::string file)
{
    _vertex = file;
}

std::string MyShader::getVertex()
{
    return _vertex;
}

void MyShader::setTransparent(bool b)
{
    _transparent = b;
}

bool MyShader::getTransparent()
{
    return _transparent;
}

void MyShader::addUniform(std::string name, int i)
{
    _intUniforms[name] = i;
}

void MyShader::addUniform(std::string name, bool b)
{
    _boolUniforms[name] = b;
}

void MyShader::addUniform(std::string name, float f)
{
    _floatUniforms[name] = f;
}

void MyShader::addUniform(std::string name, osg::Vec2 v)
{
    _vec2Uniforms[name] = v;
}

void MyShader::addUniform(std::string name, osg::Vec3 v)
{
    _vec3Uniforms[name] = v;
}

void MyShader::addUniform(std::string name, osg::Vec4 v)
{
    _vec4Uniforms[name] = v;
}

void MyShader::addUniform(std::string name, osg::Texture1D *t)
{
    _texture1DUniforms[name] = t;
}

void MyShader::addUniform(std::string name, osg::Texture2D *t)
{
    _texture2DUniforms[name] = t;
}

void MyShader::addUniform(std::string name, osg::Texture3D *t)
{
    _texture3DUniforms[name] = t;
}

void MyShader::addUniform(std::string name, osg::TextureCubeMap *t)
{
    _textureCubeUniforms[name] = t;
}

std::map<std::string, int> MyShader::getIntUniforms()
{
    return _intUniforms;
}

std::map<std::string, bool> MyShader::getBoolUniforms()
{
    return _boolUniforms;
}

std::map<std::string, float> MyShader::getFloatUniforms()
{
    return _floatUniforms;
}

std::map<std::string, osg::Vec2> MyShader::getVec2Uniforms()
{
    return _vec2Uniforms;
}

std::map<std::string, osg::Vec3> MyShader::getVec3Uniforms()
{
    return _vec3Uniforms;
}

std::map<std::string, osg::Vec4> MyShader::getVec4Uniforms()
{
    return _vec4Uniforms;
}

std::map<std::string, osg::ref_ptr<osg::Texture1D> > MyShader::getTexture1DUniforms()
{
    return _texture1DUniforms;
}

std::map<std::string, osg::ref_ptr<osg::Texture2D> > MyShader::getTexture2DUniforms()
{
    return _texture2DUniforms;
}

std::map<std::string, osg::ref_ptr<osg::Texture3D> > MyShader::getTexture3DUniforms()
{
    return _texture3DUniforms;
}

std::map<std::string, osg::ref_ptr<osg::TextureCubeMap> > MyShader::getTextureCubeUniforms()
{
    return _textureCubeUniforms;
}
