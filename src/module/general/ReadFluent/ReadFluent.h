/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30a				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *                    Headerfile for COV_READ                           *
 *									*
 ************************************************************************/

#include <api/coModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <util/DLinkList.h>
#define MAX_CELL_ZONES 1000

#define BUFSIZE 64000
#define PUTBACKSIZE 128
#define READ_CELLS 0
#define READ_FACES 1

class fluentFile
{
public:
    enum bo
    {
        LITTLE_END,
        BIG_END
    };

    fluentFile();
    ~fluentFile();
    int open(const char *fileName);
    void close();
    char getChar();
    void putBack(char c);
    int fillBuf();
    int eof();
    int getSection();
    int skipSection();
    int nextSubSection();
    int endSubSection();
    int readDez(int &);
    int readHex(int &);
    int readFloat(float &);
    void readBin(float &);
    void readBin(double &);
    void readBin(int &);
    int readString(char *, int maxlen);
    int getCurrentSection()
    {
        return currentSection;
    };

private:
    int myByteOrder_;
    char tmpBuf[1000];
    int currentSection;
    char *currentChar;
    char *lastChar;
    char buf[BUFSIZE];
    char bBuf[PUTBACKSIZE];
    int numback;
    int numOpen;
    int fd;
    int gotHeader; // did we already read the ASCII header of a section
};

const int maxNumQuads = 20;
const int maxNumTriang = 40;

class Element
{
public:
    Element();
    Element(int cell);

    //    Element(const Element& ele);

    void badInfo();
    void info()
    {
        fprintf(stderr, "Element:\n   no. of triangles: %d\n   no. of quads: %d\n", numTriangles, numQuads);
    };
    ~Element();

    // proper assignment
    //const Element& operator=(const Element& ele);

    // a tetrahedron (or hexahedron t.b.d.) is splitted from *this or an empty element is returned
    // Element simplify();
    int empty()
    {
        return empty_;
    };
    int checkTet();
    // returns true, if all faces are collected
    int addFace(int *vertices, int face, int type, int side);
    int getType(); // returns the type of element
    void setVertices(int *vl, int &numVertices); // add Vertices to the Vertex list
    void reset();

private:
    int cell_;
    int bad_;
    int empty_;

    int numTriangles;
    int numQuads;

    int triangles[maxNumTriang][3];
    int quads[maxNumQuads][4];
    int qside[maxNumQuads];
    //int tHasVertex(int t,int v) {return ((triangles[t][0] == v)||(triangles[t][1] == v)||(triangles[t][1] == v))};
    int getQuad(int v1, int v2, int notNum); //return the number of the quad (other than notNum) with the thwo vertives v1 and v2
};

class Fluent : public coModule
{

private:
    virtual int compute(const char *port);
    virtual void param(const char *, bool inMapLoading);

    // parameters
    coFileBrowserParam *p_casepath;
    coFileBrowserParam *p_datapath;
    coChoiceParam *p_data[3];
    coIntScalarParam *p_timesteps;
    coIntScalarParam *p_skip;
    coIntScalarParam *pFileIncrement;

    //ports
    coOutputPort *p_outPort1;
    coOutputPort *p_outPort2;
    coOutputPort *p_outPort3;
    coOutputPort *p_outPort4;
    coOutputPort *p_outPort5;
    coOutputPort *p_linePort;

    //  Local data
    char tmpBuf[1000];
    char *dataFileName;
    int varTypes[1000];
    int varIsFace[1000];
    int numVars;
    int numCells, numFaces, numNodes, numVertices, numElements;
    float *x_coords;
    float *y_coords;
    float *z_coords;
    int *elementTypeList;
    int *vertices;
    int *typelist;
    int *facetype;
    int *rightNeighbor;
    int *leftNeighbor;
    char *faceFlag; // whether this cell is valid (not an intersection face) which has to be ignored
    // 0 == valid
    int *cellvl;
    char *cellFlag; // whether this cell is valid or a parent cell which has to be ignored
    // 0 == valid
    int *newElementList;
    int *elemToCells_;
    int numCellZones;
    int cellZoneIds[MAX_CELL_ZONES];

    Element **Elements;
    //DLinkList<Element *> freeElements;
    Element **freeElements;
    int numFreeElements;
    int numFreeAlloc;

    int readFile(const char *fileName);
    int parseDat(const char *fileName);
    int readDat(const char *fileName, float *dest, int var, int type,
                float *dest1 = NULL, float *dest2 = NULL);
    coDistributedObject *createDataObject(const char *name, int dataSelection, char *dataFileName);
    coDistributedObject *createFaceDataObject(const char *name, int dataSelection, char *dataFileName);

    coDoLines *makeLines();

    void addVariable(int varNum);
    void addFaceVariable(int varNum);
    void updateChoice();
    void addTriangle(int face, int cell);
    void addFace(int face, int cell, int side);
    float tetraVol(float p0[3], float p1[3], float p2[3], float p3[3]);
    fluentFile file;
    int tetOnly; // Grid Contains Only Tetrahedra

public:
    Fluent(int argc, char *argv[]);
    virtual ~Fluent();
    int isTetOnly()
    {
        return tetOnly;
    };
};

static int VectorScalarMap[] = { 0, 56, 111, 115, 122 };
static const int NumVectVars = 5; // number of entries in the array above (VectorScalarMap)

static const char *FluentVecVarNames[] = {
    "CELL_DATA_NULL", //  0
    /*         "CELL_DATA_MOMENTUM(Vector)",           //  -2 2  */
    "CELL_DATA_DO_IW(Vector)", //  -1 56
    "CELL_DATA_VELOCITY(Vector)", //  -2 111
    "CELL_DATA_VELOCITY_M1(Vector)", //  -3 115
    "CELL_DATA_VELOCITY_M2(Vector)", //  -4 122
};
static const char *FluentVecFaceVarNames[] = {
    "FACE_DATA_NULL", // 0
    "FACE_DATA_DO_IW(Vector)", // -1 56
    "FACE_DATA_VELOCITY(Vector)", // -2 111
    "FACE_DATA_VELOCITY_M1(Vector)", // -3 115
    "FACE_DATA_VELOCITY_M2(Vector)", // -4 122
};
static const char *FluentVarNames[] = {
    "CELL_DATA_NULL", //  0
    "CELL_DATA_PRESSURE", //  1
    "CELL_DATA_MOMENTUM(Vector)", //  2
    "CELL_DATA_TEMPERATURE", //  3
    "CELL_DATA_ENTHALPY", //  4
    "CELL_DATA_TKE", //  5
    "CELL_DATA_TED", //  6
    "CELL_DATA_SPECIES", //  7
    "CELL_DATA_ENTHALPY", //  8
    "CELL_DATA_WSWIRL", //  9
    "CELL_DATA_DPMS_MASS", //  10
    "CELL_DATA_DPMS_MOM", //  11
    "CELL_DATA_DPMS_ENERGY", //  12
    "CELL_DATA_DPMS_SPECIES", //  13
    "CELL_DATA_DVOLUME_DT", //  14
    "CELL_DATA_BODY_FORCES", //  15
    "CELL_DATA_FMEAN", //  16
    "CELL_DATA_FVAR", //  17
    "CELL_DATA_MASS_FLUX", //  18
    "CELL_DATA_WALL_SHEAR", //  19
    "CELL_DATA_BOUNDARY_HEAT_FLUX", //  20
    "CELL_DATA_BOUNDARY_RAD_HEAT_FLUX", //  21
    "CELL_DATA_OLD_PRESSURE", //  22
    "CELL_DATA_POLLUT", //  23
    "CELL_DATA_DPMS_P1_S", //  24
    "CELL_DATA_DPMS_P1_AP", //  25
    "CELL_DATA_WALL_GAS_TEMPERATURE", //  26
    "CELL_DATA_DPMS_P1_DIFF", //  27
    "CELL_DATA_DR_SURF", //  28
    "CELL_DATA_W_M1", //  29
    "CELL_DATA_W_M2", //  30
    "CELL_DATA_DPMS_BURNOUT", //  31
    "CELL_DATA_DPMS_CONCENTRATION", //  32
    "CELL_DATA_PDF_MW", //  33
    "CELL_DATA_DPMS_WSWIRL", //  34
    "CELL_DATA_YPLUS", //  35
    "CELL_DATA_YPLUS_UTAU", //  36
    "CELL_DATA_WALL_SHEAR_SWIRL", //  37
    "CELL_DATA_WALL_T_INNER", //  38
    "CELL_DATA_POLLUT0", //  39
    "CELL_DATA_POLLUT1", //  40
    "CELL_DATA_WALL_G_INNER", //  41
    "CELL_DATA_PREMIXC", //  42
    "CELL_DATA_PREMIXC_T", //  43
    "CELL_DATA_PREMIXC_RATE", //  44
    "CELL_DATA_POLLUT2", //  45
    "CELL_DATA_POLLUT3", //  46
    "CELL_DATA_MASS_FLUX_M1", //  47
    "CELL_DATA_MASS_FLUX_M2", //  48
    "CELL_DATA_GRID_FLUX", //  49
    "CELL_DATA_DO_I", //  50
    "CELL_DATA_DO_RECON_I", //  51
    "CELL_DATA_DO_ENERGY_SOURCE", //  52
    "CELL_DATA_DO_IRRAD", //  53
    "CELL_DATA_DO_QMINUS", //  54
    "CELL_DATA_DO_IRRAD_OLD", //  55
    "CELL_DATA_DO_IWX", //  56
    "CELL_DATA_DO_IWY", //  57
    "CELL_DATA_DO_IWZ", //  58
    "CELL_DATA_MACH", //  59
    "CELL_DATA_60", //  60
    "CELL_DATA_61", //  61
    "CELL_DATA_62", //  62
    "CELL_DATA_63", //  63
    "CELL_DATA_64", //  64
    "CELL_DATA_65", //  65
    "CELL_DATA_66", //  66
    "CELL_DATA_67", //  67
    "CELL_DATA_68", //  68
    "CELL_DATA_69", //  69
    "CELL_DATA_VFLUX", //  70
    "CELL_DATA_VFLUX_M1", //  71
    "CELL_DATA_VFLUX_M2", //  72
    "CELL_DATA_73", //  73
    "CELL_DATA_74", //  74
    "CELL_DATA_75", //  75
    "CELL_DATA_76", //  76
    "CELL_DATA_77", //  77
    "CELL_DATA_78", //  78
    "CELL_DATA_79", //  79
    "CELL_DATA_80", //  80
    "CELL_DATA_81", //  81
    "CELL_DATA_82", //  82
    "CELL_DATA_83", //  83
    "CELL_DATA_84", //  84
    "CELL_DATA_85", //  85
    "CELL_DATA_86", //  86
    "CELL_DATA_87", //  87
    "CELL_DATA_88", //  88
    "CELL_DATA_89", //  89
    "CELL_DATA_90", //  90
    "CELL_DATA_91", //  91
    "CELL_DATA_92", //  92
    "CELL_DATA_93", //  93
    "CELL_DATA_94", //  94
    "CELL_DATA_95", //  95
    "CELL_DATA_96", //  96
    "CELL_DATA_97", //  97
    "CELL_DATA_98", //  98
    "CELL_DATA_99", //  99
    "CELL_DATA_100", //  100
    "CELL_DATA_DENSITY", //  101
    "CELL_DATA_MU_LAM", //  102
    "CELL_DATA_MU_TURB", //  103
    "CELL_DATA_CP", //  104
    "CELL_DATA_KTC", //  105
    "CELL_DATA_VGS_DTRM", //  106
    "CELL_DATA_VGF_DTRM", //  107
    "CELL_DATA_RSTRESS", //  108
    "CELL_DATA_THREAD_RAD_FLUX", //  109
    "CELL_DATA_SPE_Q", //  110
    "CELL_DATA_X_VELOCITY", //  111
    "CELL_DATA_Y_VELOCITY", //  112
    "CELL_DATA_Z_VELOCITY", //  113
    "CELL_DATA_NONE", //  114
    "CELL_DATA_X_VELOCITY_M1", //  115
    "CELL_DATA_Y_VELOCITY_M1", //  116
    "CELL_DATA_Z_VELOCITY_M1", //  117
    "CELL_DATA_NONE", //  118
    "CELL_DATA_TKE_M1", //  119
    "CELL_DATA_TED_M1", //  120
    "CELL_DATA_NONE", //  121
    "CELL_DATA_X_VELOCITY_M2", //  122
    "CELL_DATA_Y_VELOCITY_M2", //  123
    "CELL_DATA_Z_VELOCITY_M2", //  124
    "CELL_DATA_NONE", //  125
    "CELL_DATA_TKE_M2", //  126
    "CELL_DATA_TED_M2", //  127
    "CELL_DATA_RUU", //  128
    "CELL_DATA_RVV", //  129
    "CELL_DATA_RWW", //  130
    "CELL_DATA_RUV", //  131
    "CELL_DATA_RVW", //  132
    "CELL_DATA_RUW", //  133
    "CELL_DATA_DPMS_EROSION", //  134
    "CELL_DATA_DPMS_ACCRETION", //  135
    "CELL_DATA_FMEAN2", //  136
    "CELL_DATA_FVAR2", //  137
    "CELL_DATA_ENTHALPY_M1", //  138
    "CELL_DATA_ENTHALPY_M2", //  139
    "CELL_DATA_FMEAN_M1", //  140
    "CELL_DATA_FMEAN_M2", //  141
    "CELL_DATA_FVAR_M1", //  142
    "CELL_DATA_FVAR_M2", //  143
    "CELL_DATA_FMEAN2_M1", //  144
    "CELL_DATA_FMEAN2_M2", //  145
    "CELL_DATA_FVAR2_M1", //  146
    "CELL_DATA_FVAR2_M2", //  147
    "CELL_DATA_PREMIXC_M1", //  148
    "CELL_DATA_PREMIXC_M2", //  149
    "VOF", //  150
    "VOF_1", //  151
    "VOF_2", //  152
    "VOF_3", //  153
    "VOF_4", //  154
    "CELL_DATA_155", //  155
    "CELL_DATA_156", //  156
    "CELL_DATA_157", //  157
    "CELL_DATA_158", //  158
    "CELL_DATA_159" //  159
    "VOF_M1", //  160
    "VOF_1_M1", //  161
    "VOF_2_M1", //  162
    "VOF_3_M1", //  163
    "VOF_4_M1", //  164
    "CELL_DATA_165", //  165
    "CELL_DATA_166", //  166
    "CELL_DATA_167", //  167
    "CELL_DATA_168", //  168
    "CELL_DATA_169" //  169
    "VOF_M2", //  170
    "VOF_1_M2", //  171
    "VOF_2_M2", //  172
    "VOF_3_M2", //  173
    "VOF_4_M2", //  174
    "CELL_DATA_175", //  175
    "CELL_DATA_176", //  176
    "CELL_DATA_177", //  177
    "CELL_DATA_178", //  178
    "CELL_DATA_179" //  179
    "VOLUME_M2", //  180
    "WALL_GRID_VELOCITY", //  181
    "POLLUT7", //  182
    "POLLUT8", //  183
    "POLLUT9", //  184
    "POLLUT10", //  185
    "POLLUT11", //  186
    "POLLUT12", //  187
    "POLLUT13", //  188
    "CELL_DATA_189" //  189
    "SV_T_AUX", //  190
    "SV_T_AP_AUX", //  191
    "TOTAL_PRESSURE", //  192
    "TOTAL_TEMPERATURE", //  193
    "NRBC_DC", //  194
    "DP_TMFR", //  195
    "CELL_DATA_196", //  196
    "CELL_DATA_197", //  197
    "CELL_DATA_198", //  198
    "CELL_DATA_199" //  199
    "CELL_DATA_MASS_FRACTION", //  200
    "CELL_DATA_MASS_CONCENTRATION", //  201
    "CELL_DATA_202", //  202
    "CELL_DATA_203", //  203
    "CELL_DATA_204", //  204
    "CELL_DATA_205", //  205
    "CELL_DATA_206", //  206
    "CELL_DATA_207", //  207
    "CELL_DATA_208", //  208
    "CELL_DATA_209" //  209
    "CELL_DATA_210", //  210
    "CELL_DATA_211", //  211
    "CELL_DATA_212", //  212
    "CELL_DATA_213", //  213
    "CELL_DATA_214", //  214
    "CELL_DATA_215", //  215
    "CELL_DATA_216", //  216
    "CELL_DATA_217", //  217
    "CELL_DATA_218", //  218
    "CELL_DATA_219", //  219
    "CELL_DATA_220", //  220
    "CELL_DATA_221", //  221
    "CELL_DATA_222", //  222
    "CELL_DATA_223", //  223
    "CELL_DATA_224", //  224
    "CELL_DATA_225", //  225
    "CELL_DATA_226", //  226
    "CELL_DATA_227", //  227
    "CELL_DATA_228", //  218
    "CELL_DATA_229", //  219
    "CELL_DATA_230", //  210
    "CELL_DATA_231", //  211
    "CELL_DATA_232", //  212
    "CELL_DATA_233", //  213
    "CELL_DATA_234", //  214
    "CELL_DATA_235", //  215
    "CELL_DATA_236", //  216
    "CELL_DATA_237", //  217
    "CELL_DATA_238", //  218
    "CELL_DATA_239", //  219
    "CELL_DATA_240", //  210
    "CELL_DATA_241", //  211
    "CELL_DATA_242", //  212
    "CELL_DATA_243", //  213
    "CELL_DATA_244", //  214
    "CELL_DATA_245", //  215
    "CELL_DATA_246", //  216
    "CELL_DATA_247", //  217
    "CELL_DATA_248", //  218
    "CELL_DATA_249", //  219
    "CELL_DATA_250", //  210
    "CELL_DATA_251", //  211
    "CELL_DATA_252", //  212
    "CELL_DATA_253", //  213
    "CELL_DATA_254", //  214
    "CELL_DATA_255", //  215
    "CELL_DATA_256", //  216
    "CELL_DATA_257", //  217
    "CELL_DATA_258", //  218
    "CELL_DATA_259", //  219
    "CELL_DATA_260", //  210
    "CELL_DATA_261", //  211
    "CELL_DATA_262", //  212
    "CELL_DATA_263", //  213
    "CELL_DATA_264", //  214
    "CELL_DATA_265", //  215
    "CELL_DATA_266", //  216
    "CELL_DATA_267", //  217
    "CELL_DATA_268", //  218
    "CELL_DATA_269", //  219
    "CELL_DATA_270", //  210
    "CELL_DATA_271", //  211
    "CELL_DATA_272", //  212
    "CELL_DATA_273", //  213
    "CELL_DATA_274", //  214
    "CELL_DATA_275", //  215
    "CELL_DATA_276", //  216
    "CELL_DATA_277", //  217
    "CELL_DATA_278", //  218
    "CELL_DATA_279", //  219
    "CELL_DATA_280", //  210
    "CELL_DATA_281", //  211
    "CELL_DATA_282", //  212
    "CELL_DATA_283", //  213
    "CELL_DATA_284", //  214
    "CELL_DATA_285", //  215
    "CELL_DATA_286", //  216
    "CELL_DATA_287", //  217
    "CELL_DATA_288", //  218
    "CELL_DATA_289", //  219
    "CELL_DATA_290", //  210
    "CELL_DATA_291", //  211
    "CELL_DATA_292", //  212
    "CELL_DATA_293", //  213
    "CELL_DATA_294", //  214
    "CELL_DATA_295", //  215
    "CELL_DATA_296", //  216
    "CELL_DATA_297", //  217
    "CELL_DATA_298", //  218
    "CELL_DATA_299", //  219
    "CELL_DATA_300", //  300
    "CELL_DATA_301", //  301
    "CELL_DATA_302", //  302
    "CELL_DATA_303", //  303
    "CELL_DATA_304", //  304
    "CELL_DATA_305", //  305
    "CELL_DATA_306", //  306
    "CELL_DATA_307", //  307
    "CELL_DATA_308", //  308
    "CELL_DATA_309" //  309
};

static const char *FluentFaceVarNames[] = {
    "FACE_DATA_NULL", //  0
    "FACE_DATA_PRESSURE", //  1
    "FACE_DATA_MOMENTUM(Vector)", //  2
    "FACE_DATA_TEMPERATURE", //  3
    "FACE_DATA_ENTHALPY", //  4
    "FACE_DATA_TKE", //  5
    "FACE_DATA_TED", //  6
    "FACE_DATA_SPECIES", //  7
    "FACE_DATA_ENTHALPY", //  8
    "FACE_DATA_WSWIRL", //  9
    "FACE_DATA_DPMS_MASS", //  10
    "FACE_DATA_DPMS_MOM", //  11
    "FACE_DATA_DPMS_ENERGY", //  12
    "FACE_DATA_DPMS_SPECIES", //  13
    "FACE_DATA_DVOLUME_DT", //  14
    "FACE_DATA_BODY_FORCES", //  15
    "FACE_DATA_FMEAN", //  16
    "FACE_DATA_FVAR", //  17
    "FACE_DATA_MASS_FLUX", //  18
    "FACE_DATA_WALL_SHEAR", //  19
    "FACE_DATA_BOUNDARY_HEAT_FLUX", //  20
    "FACE_DATA_BOUNDARY_RAD_HEAT_FLUX", //  21
    "FACE_DATA_OLD_PRESSURE", //  22
    "FACE_DATA_POLLUT", //  23
    "FACE_DATA_DPMS_P1_S", //  24
    "FACE_DATA_DPMS_P1_AP", //  25
    "FACE_DATA_WALL_GAS_TEMPERATURE", //  26
    "FACE_DATA_DPMS_P1_DIFF", //  27
    "FACE_DATA_DR_SURF", //  28
    "FACE_DATA_W_M1", //  29
    "FACE_DATA_W_M2", //  30
    "FACE_DATA_DPMS_BURNOUT", //  31
    "FACE_DATA_DPMS_CONCENTRATION", //  32
    "FACE_DATA_PDF_MW", //  33
    "FACE_DATA_DPMS_WSWIRL", //  34
    "FACE_DATA_YPLUS", //  35
    "FACE_DATA_YPLUS_UTAU", //  36
    "FACE_DATA_WALL_SHEAR_SWIRL", //  37
    "FACE_DATA_WALL_T_INNER", //  38
    "FACE_DATA_POLLUT0", //  39
    "FACE_DATA_POLLUT1", //  40
    "FACE_DATA_WALL_G_INNER", //  41
    "FACE_DATA_PREMIXC", //  42
    "FACE_DATA_PREMIXC_T", //  43
    "FACE_DATA_PREMIXC_RATE", //  44
    "FACE_DATA_POLLUT2", //  45
    "FACE_DATA_POLLUT3", //  46
    "FACE_DATA_MASS_FLUX_M1", //  47
    "FACE_DATA_MASS_FLUX_M2", //  48
    "FACE_DATA_GRID_FLUX", //  49
    "FACE_DATA_DO_I", //  50
    "FACE_DATA_DO_RECON_I", //  51
    "FACE_DATA_DO_ENERGY_SOURCE", //  52
    "FACE_DATA_DO_IRRAD", //  53
    "FACE_DATA_DO_QMINUS", //  54
    "FACE_DATA_DO_IRRAD_OLD", //  55
    "FACE_DATA_DO_IWX", //  56
    "FACE_DATA_DO_IWY", //  57
    "FACE_DATA_DO_IWZ", //  58
    "FACE_DATA_MACH", //  59
    "FACE_DATA_60", //  60
    "FACE_DATA_61", //  61
    "FACE_DATA_62", //  62
    "FACE_DATA_63", //  63
    "FACE_DATA_64", //  64
    "FACE_DATA_65", //  65
    "FACE_DATA_66", //  66
    "FACE_DATA_67", //  67
    "FACE_DATA_68", //  68
    "FACE_DATA_69", //  69
    "FACE_DATA_VFLUX", //  70
    "FACE_DATA_VFLUX_M1", //  71
    "FACE_DATA_VFLUX_M2", //  72
    "FACE_DATA_73", //  73
    "FACE_DATA_74", //  74
    "FACE_DATA_75", //  75
    "FACE_DATA_76", //  76
    "FACE_DATA_77", //  77
    "FACE_DATA_78", //  78
    "FACE_DATA_79", //  79
    "FACE_DATA_80", //  80
    "FACE_DATA_81", //  81
    "FACE_DATA_82", //  82
    "FACE_DATA_83", //  83
    "FACE_DATA_84", //  84
    "FACE_DATA_85", //  85
    "FACE_DATA_86", //  86
    "FACE_DATA_87", //  87
    "FACE_DATA_88", //  88
    "FACE_DATA_89", //  89
    "FACE_DATA_90", //  90
    "FACE_DATA_91", //  91
    "FACE_DATA_92", //  92
    "FACE_DATA_93", //  93
    "FACE_DATA_94", //  94
    "FACE_DATA_95", //  95
    "FACE_DATA_96", //  96
    "FACE_DATA_97", //  97
    "FACE_DATA_98", //  98
    "FACE_DATA_99", //  99
    "FACE_DATA_100", //  100
    "FACE_DATA_DENSITY", //  101
    "FACE_DATA_MU_LAM", //  102
    "FACE_DATA_MU_TURB", //  103
    "FACE_DATA_CP", //  104
    "FACE_DATA_KTC", //  105
    "FACE_DATA_VGS_DTRM", //  106
    "FACE_DATA_VGF_DTRM", //  107
    "FACE_DATA_RSTRESS", //  108
    "FACE_DATA_THREAD_RAD_FLUX", //  109
    "FACE_DATA_SPE_Q", //  110
    "FACE_DATA_X_VELOCITY", //  111
    "FACE_DATA_Y_VELOCITY", //  112
    "FACE_DATA_Z_VELOCITY", //  113
    "FACE_DATA_NONE", //  114
    "FACE_DATA_X_VELOCITY_M1", //  115
    "FACE_DATA_Y_VELOCITY_M1", //  116
    "FACE_DATA_Z_VELOCITY_M1", //  117
    "FACE_DATA_NONE", //  118
    "FACE_DATA_TKE_M1", //  119
    "FACE_DATA_TED_M1", //  120
    "FACE_DATA_NONE", //  121
    "FACE_DATA_X_VELOCITY_M2", //  122
    "FACE_DATA_Y_VELOCITY_M2", //  123
    "FACE_DATA_Z_VELOCITY_M2", //  124
    "FACE_DATA_NONE", //  125
    "FACE_DATA_TKE_M2", //  126
    "FACE_DATA_TED_M2", //  127
    "FACE_DATA_RUU", //  128
    "FACE_DATA_RVV", //  129
    "FACE_DATA_RWW", //  130
    "FACE_DATA_RUV", //  131
    "FACE_DATA_RVW", //  132
    "FACE_DATA_RUW", //  133
    "FACE_DATA_DPMS_EROSION", //  134
    "FACE_DATA_DPMS_ACCRETION", //  135
    "FACE_DATA_FMEAN2", //  136
    "FACE_DATA_FVAR2", //  137
    "FACE_DATA_ENTHALPY_M1", //  138
    "FACE_DATA_ENTHALPY_M2", //  139
    "FACE_DATA_FMEAN_M1", //  140
    "FACE_DATA_FMEAN_M2", //  141
    "FACE_DATA_FVAR_M1", //  142
    "FACE_DATA_FVAR_M2", //  143
    "FACE_DATA_FMEAN2_M1", //  144
    "FACE_DATA_FMEAN2_M2", //  145
    "FACE_DATA_FVAR2_M1", //  146
    "FACE_DATA_FVAR2_M2", //  147
    "FACE_DATA_PREMIXC_M1", //  148
    "FACE_DATA_PREMIXC_M2", //  149
    "FACE_DATA_150", //  150
    "FACE_DATA_151", //  151
    "FACE_DATA_152", //  152
    "FACE_DATA_153", //  153
    "FACE_DATA_154", //  154
    "FACE_DATA_155", //  155
    "FACE_DATA_156", //  156
    "FACE_DATA_157", //  157
    "FACE_DATA_158", //  158
    "FACE_DATA_159" //  159
    "FACE_DATA_160", //  160
    "FACE_DATA_161", //  161
    "FACE_DATA_162", //  162
    "FACE_DATA_163", //  163
    "FACE_DATA_164", //  164
    "FACE_DATA_165", //  165
    "FACE_DATA_166", //  166
    "FACE_DATA_167", //  167
    "FACE_DATA_168", //  168
    "FACE_DATA_169" //  169
    "FACE_DATA_170", //  170
    "FACE_DATA_171", //  171
    "FACE_DATA_172", //  172
    "FACE_DATA_173", //  173
    "FACE_DATA_174", //  174
    "FACE_DATA_175", //  175
    "FACE_DATA_176", //  176
    "FACE_DATA_177", //  177
    "FACE_DATA_178", //  178
    "FACE_DATA_179" //  179
    "FACE_DATA_180", //  180
    "FACE_DATA_181", //  181
    "FACE_DATA_182", //  182
    "FACE_DATA_183", //  183
    "FACE_DATA_184", //  184
    "FACE_DATA_185", //  185
    "FACE_DATA_186", //  186
    "FACE_DATA_187", //  187
    "FACE_DATA_188", //  188
    "FACE_DATA_189" //  189
    "FACE_DATA_190", //  190
    "FACE_DATA_191", //  191
    "FACE_DATA_192", //  192
    "FACE_DATA_193", //  193
    "FACE_DATA_194", //  194
    "FACE_DATA_195", //  195
    "FACE_DATA_196", //  196
    "FACE_DATA_197", //  197
    "FACE_DATA_198", //  198
    "FACE_DATA_199" //  199
    "FACE_DATA_MASS_FRACTION", //  200
    "FACE_DATA_MASS_CONCENTRATION", //  201
    "FACE_DATA_202", //  202
    "FACE_DATA_203", //  203
    "FACE_DATA_204", //  204
    "FACE_DATA_205", //  205
    "FACE_DATA_206", //  206
    "FACE_DATA_207", //  207
    "FACE_DATA_208", //  208
    "FACE_DATA_209" //  209
    "FACE_DATA_210", //  210
    "FACE_DATA_211", //  211
    "FACE_DATA_212", //  212
    "FACE_DATA_213", //  213
    "FACE_DATA_214", //  214
    "FACE_DATA_215", //  215
    "FACE_DATA_216", //  216
    "FACE_DATA_217", //  217
    "FACE_DATA_218", //  218
    "FACE_DATA_219", //  219
    "FACE_DATA_220", //  220
    "FACE_DATA_221", //  221
    "FACE_DATA_222", //  222
    "FACE_DATA_223", //  223
    "FACE_DATA_224", //  224
    "FACE_DATA_225", //  225
    "FACE_DATA_226", //  226
    "FACE_DATA_227", //  227
    "FACE_DATA_228", //  218
    "FACE_DATA_229", //  219
    "FACE_DATA_230", //  210
    "FACE_DATA_231", //  211
    "FACE_DATA_232", //  212
    "FACE_DATA_233", //  213
    "FACE_DATA_234", //  214
    "FACE_DATA_235", //  215
    "FACE_DATA_236", //  216
    "FACE_DATA_237", //  217
    "FACE_DATA_238", //  218
    "FACE_DATA_239", //  219
    "FACE_DATA_240", //  210
    "FACE_DATA_241", //  211
    "FACE_DATA_242", //  212
    "FACE_DATA_243", //  213
    "FACE_DATA_244", //  214
    "FACE_DATA_245", //  215
    "FACE_DATA_246", //  216
    "FACE_DATA_247", //  217
    "FACE_DATA_248", //  218
    "FACE_DATA_249", //  219
    "FACE_DATA_250", //  210
    "FACE_DATA_251", //  211
    "FACE_DATA_252", //  212
    "FACE_DATA_253", //  213
    "FACE_DATA_254", //  214
    "FACE_DATA_255", //  215
    "FACE_DATA_256", //  216
    "FACE_DATA_257", //  217
    "FACE_DATA_258", //  218
    "FACE_DATA_259", //  219
    "FACE_DATA_260", //  210
    "FACE_DATA_261", //  211
    "FACE_DATA_262", //  212
    "FACE_DATA_263", //  213
    "FACE_DATA_264", //  214
    "FACE_DATA_265", //  215
    "FACE_DATA_266", //  216
    "FACE_DATA_267", //  217
    "FACE_DATA_268", //  218
    "FACE_DATA_269", //  219
    "FACE_DATA_270", //  210
    "FACE_DATA_271", //  211
    "FACE_DATA_272", //  212
    "FACE_DATA_273", //  213
    "FACE_DATA_274", //  214
    "FACE_DATA_275", //  215
    "FACE_DATA_276", //  216
    "FACE_DATA_277", //  217
    "FACE_DATA_278", //  218
    "FACE_DATA_279", //  219
    "FACE_DATA_280", //  210
    "FACE_DATA_281", //  211
    "FACE_DATA_282", //  212
    "FACE_DATA_283", //  213
    "FACE_DATA_284", //  214
    "FACE_DATA_285", //  215
    "FACE_DATA_286", //  216
    "FACE_DATA_287", //  217
    "FACE_DATA_288", //  218
    "FACE_DATA_289", //  219
    "FACE_DATA_290", //  210
    "FACE_DATA_291", //  211
    "FACE_DATA_292", //  212
    "FACE_DATA_293", //  213
    "FACE_DATA_294", //  214
    "FACE_DATA_295", //  215
    "FACE_DATA_296", //  216
    "FACE_DATA_297", //  217
    "FACE_DATA_298", //  218
    "FACE_DATA_299", //  219
    "FACE_DATA_300", //  300
    "FACE_DATA_301", //  301
    "FACE_DATA_302", //  302
    "FACE_DATA_303", //  303
    "FACE_DATA_304", //  304
    "FACE_DATA_305", //  305
    "FACE_DATA_306", //  306
    "FACE_DATA_307", //  307
    "FACE_DATA_308", //  308
    "FACE_DATA_309" //  309
};
