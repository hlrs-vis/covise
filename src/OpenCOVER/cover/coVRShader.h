/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VR_SHADER_H
#define CO_VR_SHADER_H

/*! \file
 \brief  a class which manages all uniforms, programms and statesets for a GLSL shader

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2008
         HLRS,
         Nobelstrasse 19,
         D-70550 Stuttgart,
         Germany

 \date   May 2008
 */

#include <util/coExport.h>
#include <list>
#include <map>
#include <string>
#include <osg/Shader>
#include <osg/Program>
#include <osg/Texture>
#include <osg/Drawable>

namespace covise
{
class TokenBuffer;
}

namespace osg
{
class Node;
class Geode;
}
namespace opencover
{
class coVRShader;

class COVEREXPORT coVRUniform
{

private:
    const coVRShader *shader;
    std::string name;
    std::string type;
    std::string value;
    std::string min;
    std::string max;
    std::string textureFile;
    std::string cubeMapFiles[6];
    std::string wrapMode;
    bool overwrite;
    bool unique;

public:
    const std::string &getName() const
    {
        return name;
    }
    osg::Texture::WrapMode getWrapMode() const;
    const std::string &getType() const
    {
        return type;
    }
    const std::string &getValue() const
    {
        return value;
    }
    const std::string &getMin() const
    {
        return min;
    }
    const std::string &getMax() const
    {
        return max;
    }
    const std::string &getTextureFileName() const
    {
        return textureFile;
    }
    const std::string *getCubeMapFiles() const
    {
        return cubeMapFiles;
    }
    void setMin(const std::string &m)
    {
        min = m;
    }
    void setMax(const std::string &m)
    {
        max = m;
    }
    void setOverwrite(bool o)
    {
        overwrite = o;
    }
    void setUnique(bool u)
    {
        unique = u;
    }
    void setValue(const char *value);
    void setValue(osg::Matrixd m);
    void setValue(osg::Matrixf m);
    void setValue(bool b);
    void setValue(int i);
    void setValue(float f);
    void setValue(osg::Vec3 v);
    void setValue(osg::Vec4 v);
    void setWrapMode(std::string wm);
    void setTexture(const char *textureFile, int texNum = 0);
    bool doOverwrite() const
    {
        return overwrite;
    }
    bool isUnique() const
    {
        return unique;
    }
    osg::ref_ptr<osg::Uniform> uniform;
    osg::ref_ptr<osg::Texture> texture;

    coVRUniform(const coVRShader *shader, const std::string &name, const std::string &type, const std::string &value);
    virtual ~coVRUniform();
};

class COVEREXPORT coVRAttribute
{

private:
    std::string name;
    std::string type;
    std::string value;

public:
    const std::string &getName()
    {
        return name;
    }
    const std::string &getType()
    {
        return type;
    }
    const std::string &getValue()
    {
        return value;
    }
    coVRAttribute(const std::string &name, const std::string &type, const std::string &value);
    virtual ~coVRAttribute();
};

class COVEREXPORT coVRShaderInstance
{

private:
    std::list<osg::ref_ptr<osg::Uniform> > uniforms;
    osg::Drawable *myDrawable;

public:
    coVRShaderInstance(osg::Drawable *d);
    virtual ~coVRShaderInstance();
    void addUniform(const osg::Uniform &u);
    std::list<osg::ref_ptr<osg::Uniform> > &getUniforms()
    {
        return uniforms;
    };
    osg::Uniform *getUniform(const std::string &name);
};

class COVEREXPORT coVRShader
{
    friend class coVRShaderList;

private:
    std::string name;
    std::string fileName;
    std::string dir;
    std::string defines;
    int versionMin = -1, versionMax = -1;
    std::string profile;
    bool wasCloned;
    std::list<coVRUniform *> uniforms;
    std::list<coVRAttribute *> attributes;
    std::list<coVRShaderInstance *> instances;
    osg::ref_ptr<osg::Shader> fragmentShader;
    osg::ref_ptr<osg::Shader> geometryShader;
    osg::ref_ptr<osg::Shader> vertexShader;
    osg::ref_ptr<osg::Shader> tessControlShader;
    osg::ref_ptr<osg::Shader> tessEvalShader;
    osg::ref_ptr<osg::Program> program;
    bool transparent; // the shader is transparent regardless of the users wishes
    bool opaque; // the shader is opaque regardless of the users wishes
    int geomParams[3];
    int cullFace;

public:
    std::string findAsset(const std::string &path) const;
    const std::string &getName()
    {
        return name;
    }
    bool isTransparent()
    {
        return transparent;
    };
    std::list<coVRUniform *> &getUniforms()
    {
        return uniforms;
    };
    osg::ref_ptr<osg::Shader> &getFragmentShader()
    {
        return fragmentShader;
    };
    osg::ref_ptr<osg::Shader> &getGeometryShader()
    {
        return geometryShader;
    };
    osg::ref_ptr<osg::Shader> &getVertexShader()
    {
        return vertexShader;
    };
    osg::ref_ptr<osg::Shader> &getTessControlShader()
    {
        return tessControlShader;
    };
    osg::ref_ptr<osg::Shader> &getTessEvalShader()
    {
        return tessEvalShader;
    };
    osg::ref_ptr<osg::Program> &getProgram()
    {
        return program;
    };
    int getNumVertices()
    {
        return geomParams[0];
    };
    int getInputType()
    {
        return geomParams[1];
    };
    int getOutputType()
    {
        return geomParams[2];
    };
    bool isClone()
    {
        return wasCloned;
    }
    coVRShader(const std::string &name, const std::string &d, const std::string &defines = "");
    coVRShader(const coVRShader &other);
    void setData(covise::TokenBuffer &tb);
    void setMatrixUniform(const std::string &name, osg::Matrixd m);
    void setMatrixUniform(const std::string &name, osg::Matrixf m);
    void setBoolUniform(const std::string &name, bool b);
    void setIntUniform(const std::string &name, int i);
    void setFloatUniform(const std::string &name, float f);
    void setVec3Uniform(const std::string &name, osg::Vec3 v);
    void setVec4Uniform(const std::string &name, osg::Vec4 v);
    void setNumVertices(int);
    void setInputType(int);
    void setOutputType(int);
    osg::Uniform *getUniform(const std::string &name);
    coVRUniform *getcoVRUniform(const std::string &name);

    //	  void remove(osg::Node *);
    coVRShaderInstance *apply(osg::Node *);
    void apply(osg::StateSet *);
    coVRShaderInstance *apply(osg::Geode *geode, osg::Drawable *drawable);
    void setUniformesFromAttribute(const char *uniformValues);

    void storeMaterial();
    void loadMaterial();

    virtual ~coVRShader();
};

class COVEREXPORT coVRShaderList : public std::list<coVRShader *>
{
private:
    static coVRShaderList *s_instance;
    coVRShaderList();
    void loadMaterials();
    std::vector<osg::ref_ptr<osg::Uniform>> lightEnabled;
    osg::ref_ptr<osg::Uniform> timeUniform;
    osg::ref_ptr<osg::Uniform> timeStepUniform;
    osg::ref_ptr<osg::Uniform> lightMatrix;
    osg::ref_ptr<osg::Uniform> projectionMatrix; //neue Projektionsmatrix
    osg::ref_ptr<osg::Uniform> viewMatrix;
    osg::ref_ptr<osg::Uniform> durationUniform;
    osg::ref_ptr<osg::Uniform> viewportWidthUniform;
    osg::ref_ptr<osg::Uniform> viewportHeightUniform;
    osg::ref_ptr<osg::Uniform> stereoUniform; // 0 = LEFT, 1 = RIGHT
    void applyParams(coVRShader *shader, std::map<std::string, std::string> *params);
    std::map<std::string,osg::Uniform*> globalUniforms;
    std::pair<int, int> glslVersionRange{-1, -1};

public:
    ~coVRShaderList();
    coVRShader *get(const std::string &name, std::map<std::string, std::string> *params = NULL);
    coVRShader *getUnique(const std::string &n, std::map<std::string, std::string> *params = NULL, const std::string &defines = "");
    coVRShader *add(const std::string &name, const std::string &dirName, const std::string &defines="");
    static coVRShaderList *instance();
    void setData(covise::TokenBuffer &tb);
    void addGlobalUniform(const std::string &, osg::Uniform *);
    void removeGlobalUniform(osg::Uniform*);
    osg::Uniform* getGlobalUniform(const std::string&);
    osg::Uniform* getLightEnabled(size_t lightnum);
    osg::Uniform *getTime();
    osg::Uniform *getTimeStep();
    osg::Uniform *getLight();
    osg::Uniform *getProjection(); // neue Projektionsmatrix
    osg::Uniform *getView();
    osg::Uniform *getDuration();
    osg::Uniform *getViewportWidth();
    osg::Uniform *getViewportHeight();
    osg::Uniform *getStereo();
    void update();

    void init(osg::GLExtensions *glext = nullptr);
    void remove(osg::Node *);

    std::pair<int, int> glslVersion() const;
};

class COVEREXPORT ShaderNode : public osg::Drawable
{

public:
    enum StereoView
    {
        Left = 128,
        Right = 256
    };
    ShaderNode(StereoView v);
    virtual ~ShaderNode();
    static ShaderNode *theNode;
    virtual void drawImplementation(osg::RenderInfo &renderInfo) const;
    /** Clone the type of an object, with Object* return type.
	Must be defined by derived classes.*/
    virtual osg::Object *cloneType() const;

    /** Clone the an object, with Object* return type.
	Must be defined by derived classes.*/
    virtual osg::Object *clone(const osg::CopyOp &) const;
    StereoView view;

private:
};



/**
The coTangentSpaceGenerator class generates three arrays containing tangent-space basis vectors.
It takes a texture-mapped Geometry object as input, traverses its primitive sets and computes
Tangent, Normal and Binormal vectors for each vertex, storing them into arrays.
The resulting arrays can be used as vertex program varying (per-vertex) parameters,
enabling advanced effects like bump-mapping.
To use this class, simply call the generate() method specifying the Geometry object
you want to process;
then you can retrieve the TBN arrays by calling getTangentArray(), getNormalArray()
and getBinormalArray() methods.
*/
class  coTangentSpaceGenerator : public osg::Referenced {
public:
	coTangentSpaceGenerator();
	coTangentSpaceGenerator(const coTangentSpaceGenerator &copy, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);

	void generate(osg::Geometry *geo);

	inline osg::Vec4Array *getTangentArray() { return T_.get(); }
	inline const osg::Vec4Array *getTangentArray() const { return T_.get(); }
	inline void setTangentArray(osg::Vec4Array *array) { T_ = array; }

	inline osg::Vec4Array *getNormalArray() { return N_.get(); }
	inline const osg::Vec4Array *getNormalArray() const { return N_.get(); }
	inline void setNormalArray(osg::Vec4Array *array) { N_ = array; }

	inline osg::Vec4Array *getBinormalArray() { return B_.get(); }
	inline const osg::Vec4Array *getBinormalArray() const { return B_.get(); }
	inline void setBinormalArray(osg::Vec4Array *array) { B_ = array; }

	inline osg::IndexArray *getIndices() { return indices_.get(); }

protected:

	virtual ~coTangentSpaceGenerator() {}
	coTangentSpaceGenerator &operator=(const coTangentSpaceGenerator &) { return *this; }

	void compute(osg::PrimitiveSet *pset,
		const osg::Array *vx,
		const osg::Array *nx,
		int iA, int iB, int iC);

	osg::ref_ptr<osg::Vec4Array> T_;
	osg::ref_ptr<osg::Vec4Array> B_;
	osg::ref_ptr<osg::Vec4Array> N_;
	osg::ref_ptr<osg::UIntArray> indices_;
};


}
#endif
