#ifndef OSG_UTIL_H
#define OSG_UTIL_H

#ifndef __gl_h_
#include <GL/glew.h>
#endif

#include <cover/coVRConfig.h>
#include <osg/StateAttribute>
#include <osg/Vec3>
#include <osg/StateSet>
#include <osg/GraphicsContext>
#include <osg/Uniform>
#include <osg/Node>
#include <osg/Drawable>



namespace osg_util {

    // Dump-Funktionen
    void dumpStateSet(const osg::StateSet* ss);
    void dumpStateSetOld(const osg::StateSet* ss);
    void dumpStateSetWithInheritance(const osg::StateSet* ss);
    void dumpAllModes(const osg::StateSet* ss);
    void dumpInheritedAttributes(const osg::StateSet* ss, int depth, std::set<const osg::StateSet*>& visited);

    // OpenGL-Debugging
    void waitForOpenGLContext();
    void printAllExistingVAOs(GLuint maxID = 100);
    void printCandidateVAO(GLuint candidate);

    // GraphicsContext & Window
    void printGraphicsContextAttributes(const osg::GraphicsContext* gc);
    void printAllGraphicsContextsAndWindows();

    // COver-Konfiguration
    void printPipeStruct(const opencover::pipeStruct& p);
    void printCoVRConfigOverview();
    void printBlendingTextureStruct(const opencover::blendingTextureStruct& bt);
    void printViewportStruct(const opencover::viewportStruct& vp);
    void printWindowStruct(const opencover::windowStruct& w);
    void printAngleStruct(const opencover::angleStruct& a);
    void printPBOStruct(const opencover::PBOStruct& pbo);
    void printChannelStruct(const opencover::channelStruct& ch);
    void printScreenStruct(const opencover::screenStruct& s);

    void dumpAllStateAttributes(const osg::StateSet* ss);
    std::string vec3ToString(const osg::Vec3& v);
    std::string matrixdToString(const osg::Matrixd& m);

    void wait_for_opengl_context();

    static const char* overrideValueToString(osg::StateAttribute::OverrideValue ov)
    {
        switch (ov)
        {
        case osg::StateAttribute::OFF:       return "OFF";
        case osg::StateAttribute::ON:        return "ON";
        case osg::StateAttribute::OVERRIDE:  return "OVERRIDE";
        default:                             return "UNKNOWN";
        }
    }


    static const char* modeValueToString(osg::StateAttribute::GLModeValue mv)
    {
        switch (mv)
        {
        case osg::StateAttribute::ON:      return "ON";
        case osg::StateAttribute::OFF:     return "OFF";
        default:                           return "INHERIT";
        }
    }

    static const std::vector<GLenum> ALL_GL_MODES = {
        GL_DEPTH_TEST,
        GL_BLEND,
        GL_ALPHA_TEST,
        GL_CULL_FACE,
        GL_STENCIL_TEST,
        GL_SCISSOR_TEST,
        GL_POLYGON_OFFSET_FILL,
        GL_RESCALE_NORMAL,
        GL_NORMALIZE,
        GL_COLOR_MATERIAL,
        GL_LINE_SMOOTH,
        GL_POLYGON_SMOOTH,
        GL_TEXTURE_1D,
        GL_TEXTURE_2D,
        GL_POINT_SMOOTH,
        GL_MAP1_VERTEX_3,
        GL_MAP1_VERTEX_4
    };

    // Values can represent either GLModeValue or OverrideValue using bitwise flags
    struct Values {
        enum E {
            OFF = 0,
            ON = 1 << 0,  // 0x1
            OVERRIDE = 1 << 1,  // 0x2
            PROTECTED = 1 << 2,  // 0x4
            INHERIT = 1 << 3   // 0x8
        };

        friend inline E operator|(E lhs, E rhs) {
            return static_cast<E>(static_cast<int>(lhs) | static_cast<int>(rhs));
        }
        friend inline E& operator|=(E& lhs, E rhs) {
            lhs = lhs | rhs;
            return lhs;
        }
        friend inline E operator&(E lhs, E rhs) {
            return static_cast<E>(static_cast<int>(lhs) & static_cast<int>(rhs));
        }
        friend inline bool any(E v) {
            return v != OFF;
        }
    };


    static const std::vector<std::pair<int, const char*>> ATTRIBUTE_TYPE_NAMES = {
        { 0,  "TEXTURE"                },
        { 1,  "POLYGONMODE"            },
        { 2,  "POLYGONOFFSET"          },
        { 3,  "MATERIAL"               },
        { 4,  "ALPHAFUNC"              },
        { 5,  "ANTIALIAS"              },
        { 6,  "COLORTABLE"             },
        { 7,  "CULLFACE"               },
        { 8,  "FOG"                    },
        { 9,  "FRONTFACE"              },
        {10,  "LIGHT"                  },
        {11,  "POINT"                  },
        {12,  "LINEWIDTH"              },
        {13,  "LINESTIPPLE"            },
        {14,  "POLYGONSTIPPLE"         },
        {15,  "SHADEMODEL"             },
        {16,  "TEXENV"                 },
        {17,  "TEXENVFILTER"           },
        {18,  "TEXGEN"                 },
        {19,  "TEXMAT"                 },
        {20,  "LIGHTMODEL"             },
        {21,  "BLENDFUNC"              },
        {22,  "BLENDEQUATION"          },
        {23,  "LOGICOP"                },
        {24,  "STENCIL"                },
        {25,  "COLORMASK"              },
        {26,  "DEPTH"                  },
        {27,  "VIEWPORT"               },
        {28,  "SCISSOR"                },
        {29,  "BLENDCOLOR"             },
        {30,  "MULTISAMPLE"            },
        {31,  "CLIPPLANE"              },
        {32,  "COLORMATRIX"            },
        {33,  "VERTEXPROGRAM"          },
        {34,  "FRAGMENTPROGRAM"        },
        {35,  "POINTSPRITE"            },
        {36,  "PROGRAM"                },
        {37,  "CLAMPCOLOR"             },
        {38,  "HINT"                   },
        {39,  "SAMPLEMASKI"            },
        {40,  "PRIMITIVERESTARTINDEX"  },
        {41,  "CLIPCONTROL"            },
        {42,  "VALIDATOR"              },
        {43,  "VIEWMATRIXEXTRACTOR"    },
        {44,  "UNIFORMBUFFERBINDING"   },
        {45,  "TRANSFORMFEEDBACKBUFFERBINDING"},
        {46,  "ATOMICCOUNTERBUFFERBINDING"},
        {47,  "PATCH_PARAMETER"        },
        {48,  "FRAME_BUFFER_OBJECT"    },
        {49,  "VERTEX_ATTRIB_DIVISOR"  },
        {50,  "SHADERSTORAGEBUFFERBINDING"},
        {51,  "INDIRECTDRAWBUFFERBINDING"},
        {52,  "VIEWPORTINDEXED"        },
        {53,  "DEPTHRANGEINDEXED"      },
        {54,  "SCISSORINDEXED"         },
        {55,  "BINDIMAGETEXTURE"       },
        {56,  "SAMPLER"                },
        {100, "CAPABILITY"             }
    };

}


#endif // OSG_UTIL_H