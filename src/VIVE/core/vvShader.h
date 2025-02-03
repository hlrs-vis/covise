/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
#include <util/coExport.h>
#include <list>
#include <map>
#include <string>
#include <vsg/maths/vec3.h>
#include <vsg/maths/mat4.h>
#include <vsg/nodes/Node.h>

namespace covise
{
class TokenBuffer;
}

namespace vive
{
class vvShader;

class VVCORE_EXPORT coVRUniform
{

private:
    const vvShader *shader;
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
    //VkSamplerAddressMode getWrapMode() const;
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
    void setValue(vsg::dmat4 m);
    void setValue(vsg::mat4 m);
    void setValue(bool b);
    void setValue(float f);
    void setValue(vsg::vec3 v);
    void setValue(vsg::vec4 v);
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
    //vsg::ref_ptr<osg::Uniform> uniform;
    //vsg::ref_ptr<osg::Texture> texture;

    coVRUniform(const vvShader *shader, const std::string &name, const std::string &type, const std::string &value);
    virtual ~coVRUniform();
};

class VVCORE_EXPORT coVRAttribute
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

class VVCORE_EXPORT vvShaderInstance
{

private:
    //std::list<vsg::ref_ptr<osg::Uniform> > uniforms;
    vsg::Node *myDrawable;

public:
    vvShaderInstance(vsg::Node *d);
    virtual ~vvShaderInstance();
    /*void addUniform(const osg::Uniform& u);
    std::list<vsg::ref_ptr<osg::Uniform> > &getUniforms()
    {
        return uniforms;
    };
    osg::Uniform *getUniform(const std::string &name);*/
};

class VVCORE_EXPORT vvShader
{
    friend class vvShaderList;

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
    std::list<vvShaderInstance *> instances;
    /*vsg::ref_ptr<osg::Shader> fragmentShader;
    vsg::ref_ptr<osg::Shader> geometryShader;
    vsg::ref_ptr<osg::Shader> vertexShader;
    vsg::ref_ptr<osg::Shader> tessControlShader;
    vsg::ref_ptr<osg::Shader> tessEvalShader;
    vsg::ref_ptr<osg::Program> program;*/
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
    /*vsg::ref_ptr<osg::Shader>& getFragmentShader()
    {
        return fragmentShader;
    };
    vsg::ref_ptr<osg::Shader> &getGeometryShader()
    {
        return geometryShader;
    };
    vsg::ref_ptr<osg::Shader> &getVertexShader()
    {
        return vertexShader;
    };
    vsg::ref_ptr<osg::Shader> &getTessControlShader()
    {
        return tessControlShader;
    };
    vsg::ref_ptr<osg::Shader> &getTessEvalShader()
    {
        return tessEvalShader;
    };
    vsg::ref_ptr<osg::Program> &getProgram()
    {
        return program;
    };*/
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
    vvShader(const std::string &name, const std::string &d, const std::string &defines = "");
    vvShader(const vvShader &other);
    void setData(covise::TokenBuffer &tb);
    void setMatrixUniform(const std::string &name, vsg::dmat4 m);
    void setMatrixUniform(const std::string &name, vsg::mat4 m);
    void setBoolUniform(const std::string &name, bool b);
    void setFloatUniform(const std::string &name, float f);
    void setVec3Uniform(const std::string &name, vsg::vec3 v);
    void setVec4Uniform(const std::string &name, vsg::vec4 v);
    void setNumVertices(int);
    void setInputType(int);
    void setOutputType(int);
    //osg::Uniform *getUniform(const std::string &name);
    //	  void remove(vsg::Node *);
    vvShaderInstance *apply(vsg::Node *);
    void setUniformesFromAttribute(const char *uniformValues);

    void storeMaterial();
    void loadMaterial();

    virtual ~vvShader();
};

class VVCORE_EXPORT vvShaderList : public std::list<vvShader *>
{
private:
    static vvShaderList *s_instance;
    vvShaderList();
    void loadMaterials();
    /*std::vector<vsg::ref_ptr<osg::Uniform>> lightEnabled;
    vsg::ref_ptr<osg::Uniform> timeUniform;
    vsg::ref_ptr<osg::Uniform> timeStepUniform;
    vsg::ref_ptr<osg::Uniform> lightMatrix;
    vsg::ref_ptr<osg::Uniform> projectionMatrix; //neue Projektionsmatrix
    vsg::ref_ptr<osg::Uniform> viewMatrix;
    vsg::ref_ptr<osg::Uniform> durationUniform;
    vsg::ref_ptr<osg::Uniform> viewportWidthUniform;
    vsg::ref_ptr<osg::Uniform> viewportHeightUniform;
    vsg::ref_ptr<osg::Uniform> stereoUniform; // 0 = LEFT, 1 = RIGHT*/
    void applyParams(vvShader *shader, std::map<std::string, std::string> *params);
    //std::map<std::string,osg::Uniform*> globalUniforms;
    std::pair<int, int> glslVersionRange{-1, -1};

public:
    ~vvShaderList();
    vvShader *get(const std::string &name, std::map<std::string, std::string> *params = NULL);
    vvShader *getUnique(const std::string &n, std::map<std::string, std::string> *params = NULL, const std::string &defines = "");
    vvShader *add(const std::string &name, const std::string &dirName, const std::string &defines="");
    static vvShaderList *instance();
    void setData(covise::TokenBuffer &tb);
    /*void addGlobalUniform(const std::string&, osg::Uniform*);
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
    osg::Uniform *getStereo();*/
    void update();

    void init();
    void remove(vsg::Node *);

    std::pair<int, int> glslVersion() const;
};

class VVCORE_EXPORT ShaderNode : public vsg::Node
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
    //virtual void drawImplementation(osg::RenderInfo &renderInfo) const;
    /** Clone the type of an object, with Object* return type.
	Must be defined by derived classes.*/
   // virtual osg::Object *cloneType() const;

    /** Clone the an object, with Object* return type.
	Must be defined by derived classes.*/
   // virtual osg::Object *clone(const osg::CopyOp &) const;
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
/*
class  coTangentSpaceGenerator : public osg::Referenced {
public:
	coTangentSpaceGenerator();
	coTangentSpaceGenerator(const coTangentSpaceGenerator &copy, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);

	void generate(vsg::Node *geo);

	inline vsg::vec4Array *getTangentArray() { return T_.get(); }
	inline const vsg::vec4Array *getTangentArray() const { return T_.get(); }
	inline void setTangentArray(vsg::vec4Array *array) { T_ = array; }

	inline vsg::vec4Array *getNormalArray() { return N_.get(); }
	inline const vsg::vec4Array *getNormalArray() const { return N_.get(); }
	inline void setNormalArray(vsg::vec4Array *array) { N_ = array; }

	inline vsg::vec4Array *getBinormalArray() { return B_.get(); }
	inline const vsg::vec4Array *getBinormalArray() const { return B_.get(); }
	inline void setBinormalArray(vsg::vec4Array *array) { B_ = array; }

	inline osg::IndexArray *getIndices() { return indices_.get(); }

protected:

	virtual ~coTangentSpaceGenerator() {}
	coTangentSpaceGenerator &operator=(const coTangentSpaceGenerator &) { return *this; }

	void compute(osg::PrimitiveSet *pset,
		const osg::Array *vx,
		const osg::Array *nx,
		int iA, int iB, int iC);

	vsg::ref_ptr<vsg::vec4Array> T_;
	vsg::ref_ptr<vsg::vec4Array> B_;
	vsg::ref_ptr<vsg::vec4Array> N_;
	vsg::ref_ptr<osg::UIntArray> indices_;
};
*/

}
