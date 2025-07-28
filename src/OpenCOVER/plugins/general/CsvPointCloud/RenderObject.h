#ifndef COVISE_CSV_POINT_CLOUD_RENDER_OBJ
#define COVISE_CSV_POINT_CLOUD_RENDER_OBJ

#include <cover/RenderObject.h>
#include <PluginUtil/colors/coColorMap.h>

class CsvRenderObject : public opencover::RenderObject
{
public:
    const char *getName() const override { return "CsvPointCloud"; }
    void setObjName(const std::string &name);
    bool isGeometry() const override { return false; }
    RenderObject *getGeometry() const override { return nullptr; }
    RenderObject *getNormals() const override { return nullptr; }
    RenderObject *getColors() const override { return nullptr; }
    RenderObject *getTexture() const override { return nullptr; }
    RenderObject *getVertexAttribute() const override { return nullptr; }
    RenderObject *getColorMap(int idx) const override { return nullptr; }

    const char *getAttribute(const char *) const override;

    //XXX: hacks for Volume plugin and Tracer
    bool isSet() const override { return false; }
    size_t getNumElements() const override { return 0; }
    RenderObject *getElement(size_t idx) const override { return nullptr; }

    bool isUniformGrid() const override { return false; }
    void getSize(int &nx, int &ny, int &nz) const override { nx = 0; ny = 0; nz = 0; }
    float getMin(int channel) const override { return 0; }
    float getMax(int channel) const override { return 0; }
    void getMinMax(float &xmin, float &xmax,
                           float &ymin, float &ymax,
                           float &zmin, float &zmax) const override { }

    bool isVectors() const override { return false; }
    const unsigned char *getByte(opencover::Field::Id idx) const override { return nullptr; }
    const int *getInt(opencover::Field::Id idx) const override { return nullptr; }
    const float *getFloat(opencover::Field::Id idx) const override { return nullptr; }

    bool isUnstructuredGrid() const override { return false; }
    bool fromCovise() const override { return true; }

private:
    std::string m_objName;
};

#endif //COVISE_CSV_POINT_CLOUD_RENDER_OBJ
