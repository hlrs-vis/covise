#ifndef WRITENETCDF_H
#define WRITENETCDF_H



#include <api/coSimpleModule.h>
#include <netcdfcpp.h>

#define numVars 3

using namespace covise;


class WriteNetCDF: public coSimpleModule
{
private:
    coInputPort *p_gridIn, *p_dataIn[numVars];
    coFileBrowserParam *p_fileName;
    NcFile *ncOutFile;

    coStringParam *p_varName[numVars];


    virtual int compute(const char *port);
  //  void write(/*netcdf *obj*/,const char *filename);

    float *x_c, *y_c, *z_c;
public:
    WriteNetCDF(int argc, char *argv[]);
    virtual ~WriteNetCDF();
};
















#endif
