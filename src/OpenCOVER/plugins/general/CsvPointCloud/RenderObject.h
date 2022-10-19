#ifndef COVISE_CSV_POINT_CLOUD_RENDER_OBJ
#define COVISE_CSV_POINT_CLOUD_RENDER_OBJ

#include <cover/RenderObject.h>
#include <PluginUtil/coColorMap.h>

class CsvRenderObject : public opencover::RenderObject
{
public:
    const char *getName() const { return "CsvPointCloud"; }
    void setObjName(const std::string &name);
    bool isGeometry() const { return false; }
    RenderObject *getGeometry() const{ return nullptr; }
    RenderObject *getNormals() const{ return nullptr; }
    RenderObject *getColors() const{ return nullptr; }
    RenderObject *getTexture() const{ return nullptr; }
    RenderObject *getVertexAttribute() const{ return nullptr; }
    RenderObject *getColorMap(int idx) const{ return nullptr; }

    const char *getAttribute(const char *) const override;

    //XXX: hacks for Volume plugin and Tracer
    bool isSet() const{ return false; }
    size_t getNumElements() const{ return 0; }
    RenderObject *getElement(size_t idx) const{ return nullptr; }

    bool isUniformGrid() const{ return false; }
    void getSize(int &nx, int &ny, int &nz) const{}
    float getMin(int channel) const{ return 0; }
    float getMax(int channel) const{ return 0; }
    void getMinMax(float &xmin, float &xmax,
                           float &ymin, float &ymax,
                           float &zmin, float &zmax) const{ }

    bool isVectors() const{ return false; }
    const unsigned char *getByte(opencover::Field::Id idx) const{ return nullptr; }
    const int *getInt(opencover::Field::Id idx) const{ return nullptr; }
    const float *getFloat(opencover::Field::Id idx) const{ return nullptr; }

    bool isUnstructuredGrid() const{ return false; }
private:
    std::string m_objName;
};

#endif //COVISE_CSV_POINT_CLOUD_RENDER_OBJ