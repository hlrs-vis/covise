// WriteUnstructured
// Filip Sadlo 2007
// Computer Graphics Laboratory, ETH Zurich

#include "api/coSimpleModule.h"

#include "covise_ext.h"
std::vector<myParam *> myParams;

class myModule : public coSimpleModule
{
private:
    // Member functions
    virtual void postInst();
    virtual void param(const char *name, bool);
    virtual int compute(const char *);

    // Inputs
    coInputPort *grid;
    coInputPort *scalar;
    coInputPort *vector0;
    coInputPort *vector1;

    // Outputs

    // Parameters
    coFileBrowserParam *fileName;
    coBooleanParam *sequenceName;

public:
    myModule(int argc, char **argv)
        : coSimpleModule(argc, argv, "WriteUnstructured")
    {
        grid = addInputPort("grid", "UnstructuredGrid", "Unstructured Grid");
        vector0 = addInputPort("vector0", "Vec3", "Vector Data");
        vector0->setRequired(0);
        vector1 = addInputPort("vector1", "Vec3", "Vector Data");
        vector1->setRequired(0);
        scalar = addInputPort("scalar", "Float", "Scalar Data");
        scalar->setRequired(0);

        fileName = addFileBrowserParam("fileName", "name for unstructured output");
        sequenceName = addBooleanParam("sequenceName", "for sequential name");

        // default values
        sequenceName->setValue(false);
    }

    virtual ~myModule()
    {
    }
};
