/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MY_SHADER_H
#define MY_SHADER_H

#include <osg/Vec4>
#include <osg/Texture1D>
#include <osg/Texture2D>
#include <osg/Texture3D>
#include <osg/TextureCubeMap>

//Shader
class MyShader
{
public:
    MyShader();
    ~MyShader();

    void setFragmentFile(std::string file);
    std::string getFragment();
    void setVertexFile(std::string file);
    std::string getVertex();
    void setTransparent(bool b);
    bool getTransparent();
    void addUniform(std::string name, int i);
    void addUniform(std::string name, bool b);
    void addUniform(std::string name, float f);
    void addUniform(std::string name, osg::Vec2 v);
    void addUniform(std::string name, osg::Vec3 v);
    void addUniform(std::string name, osg::Vec4 v);
    void addUniform(std::string name, osg::Texture1D *t);
    void addUniform(std::string name, osg::Texture2D *t);
    void addUniform(std::string name, osg::Texture3D *t);
    void addUniform(std::string name, osg::TextureCubeMap *t);
    std::map<std::string, int> getIntUniforms();
    std::map<std::string, bool> getBoolUniforms();
    std::map<std::string, float> getFloatUniforms();
    std::map<std::string, osg::Vec2> getVec2Uniforms();
    std::map<std::string, osg::Vec3> getVec3Uniforms();
    std::map<std::string, osg::Vec4> getVec4Uniforms();
    std::map<std::string, osg::ref_ptr<osg::Texture1D> > getTexture1DUniforms();
    std::map<std::string, osg::ref_ptr<osg::Texture2D> > getTexture2DUniforms();
    std::map<std::string, osg::ref_ptr<osg::Texture3D> > getTexture3DUniforms();
    std::map<std::string, osg::ref_ptr<osg::TextureCubeMap> > getTextureCubeUniforms();

private:
    std::string _fragment;
    std::string _vertex;
    bool _transparent;
    std::map<std::string, int> _intUniforms;
    std::map<std::string, bool> _boolUniforms;
    std::map<std::string, float> _floatUniforms;
    std::map<std::string, osg::Vec2> _vec2Uniforms;
    std::map<std::string, osg::Vec3> _vec3Uniforms;
    std::map<std::string, osg::Vec4> _vec4Uniforms;
    std::map<std::string, osg::ref_ptr<osg::Texture1D> > _texture1DUniforms;
    std::map<std::string, osg::ref_ptr<osg::Texture2D> > _texture2DUniforms;
    std::map<std::string, osg::ref_ptr<osg::Texture3D> > _texture3DUniforms;
    std::map<std::string, osg::ref_ptr<osg::TextureCubeMap> > _textureCubeUniforms;
};

#endif
