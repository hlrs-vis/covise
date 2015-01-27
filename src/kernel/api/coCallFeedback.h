/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCALLFEEDBACK_H
#define COCALLFEEDBACK_H

#include <appl/CoviseBase.h>
#include "coModule.h"

namespace covise
{

class coCallFeedback : public CoviseBase
{
public:
    coCallFeedback();
    coCallFeedback(const coCallFeedback &orig);
    virtual ~coCallFeedback();
    void init(const char *n, const char *attrib);
    void unMask(char *string) const;
    void sendFeedback(const char *info, const char *key, const char *data = NULL);
    void executeModule(void);
    void copyModule();
    void copyModuleExec();
    void deleteModule();

    int getBooleanParam(unsigned int paraNo, int &value) const;
    int getIntScalarParam(unsigned int paraNo, int &value) const;
    int getFloatScalarParam(unsigned int paraNo, float &value) const;
    int getIntSliderParam(unsigned int paraNo, int &min, int &max, int &val) const;
    int getFloatSliderParam(unsigned int paraNo, float &min, float &max, float &val) const;
    int getIntVectorParam(unsigned int paraNo, int &numElem, int *&val) const;
    int getFloatVectorParam(unsigned int paraNo, int &numElem, float *&val) const;
    int getStringParam(unsigned int paraNo, const char *&val) const;
    int getChoiceParam(unsigned int paraNo, int &num, char **&labels, int &active) const;
    int getFileBrowserParam(unsigned int paraNo, char *&val) const;
    int getBooleanParam(const std::string &paraName, int &value) const;
    int getIntScalarParam(const std::string &paraName, int &value) const;
    int getFloatScalarParam(const std::string &paraName, float &value) const;
    int getIntSliderParam(const std::string &paraName, int &min, int &max, int &val) const;
    int getFloatSliderParam(const std::string &paraName, float &min, float &max, float &val) const;
    int getIntVectorParam(const std::string &paraName, int &numElem, int *&val) const;
    int getFloatVectorParam(const std::string &paraName, int &numElem, float *&val) const;
    int getStringParam(const std::string &paraName, const char *&val) const;
    int getChoiceParam(const std::string &paraName, int &num, char **&labels, int &active) const;
    int getFileBrowserParam(const std::string &paraName, char *&val) const;
    void setBooleanParam(const char *name, int val);
    void setScalarParam(const char *name, float val);
    void setScalarParam(const char *name, int val);
    void setSliderParam(const char *name, float min, float max, float value);
    void setSliderParam(const char *name, int min, int max, int value);
    void setVectorParam(const char *name, int numElem, float *field);
    void setVectorParam(const char *name, float u, float v, float w);
    void setVectorParam(const char *name, int numElem, int *field);
    void setVectorParam(const char *name, int u, int v, int w);
    void setStringParam(const char *name, const char *val);
    void setChoiceParam(const char *name, int num, const char *const *list, int pos);
    void setFileBrowserParam(const char *name, const char *val);

    void setChoiceParam(const char *name, int const pos);
    void setChoiceParam(const char *name, std::string const label);
    void setFloatSliderParam(const char *name, float const value);

    const char *getObjName()
    {
        return d_objName;
    }
    const char *getPluginName()
    {
        return d_pluginName;
    }
    const char *getModuleName()
    {
        return d_moduleName;
    }
    int getModuleInstance()
    {
        return d_moduleInstance;
    }
    const char *getModuleHost()
    {
        return d_moduleHost;
    }
    const char *getInfo()
    {
        return d_infoString;
    }
    int getNumParam() const
    {
        return d_numPara;
    }
    int getNumUser() const
    {
        return d_numUser;
    }
    const char *getParaName(unsigned int i) const;
    const char *getParaType(unsigned int i) const;
    const char *getParaValue(unsigned int i) const;
    const char *getString(unsigned int i) const;
    void print(FILE *outfile);

private:
    struct paraRec
    {
        const char *name;
        const char *type;
        const char *value;
    };

    char *d_feedbackInfo; // the part of the attrib created by ApplLib
    int d_refCount;
    char *d_objName; // the name of the DO with the attirbs
    char *d_moduleName; // the name of the module
    int d_moduleInstance; // the instance of the module
    char *d_moduleHost; // the hostname of the module
    char *d_pluginName; // the name of the Plugin to load for this Interactor
    char *d_infoString; // whatever the user put in the attribute

    unsigned int d_numPara; // number of parameters
    unsigned int d_numUser; // number of user strings
    paraRec *d_param; // Parameters
    char **d_user; // User strings
    char *d_splitString; // save splitted string in here
};
}
#endif /* COCALLFEEDBACK_H */
