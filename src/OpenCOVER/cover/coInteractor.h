/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_INTERACTOR_H
#define CO_INTERACTOR_H

/*! \file
 \brief  feedback parameter changes to COVISE

 \author
 \author (C)
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <util/coExport.h>

#include <cstring>
#include <cstdio>
#include <string>
#include <osg/Referenced>

namespace opencover
{
class RenderObject;

//! abstract feedback class for interacting with parameters of visualization modules (e.g. COVISE or Vistle)
class COVEREXPORT coInteractor
{
public:
    coInteractor();
    virtual ~coInteractor();

    //! return no. of users
    int refCount() const;

    //! if you get an interactor and you want to keep it use ...
    void incRefCount();

    //! if you don't need the interactor any more use ...
    void decRefCount();

    virtual void removedObject() = 0;

    /// returns true, if Interactor comes from same Module as intteractor i;
    virtual bool isSameModule(coInteractor *i) const = 0;

    /// returns true, if Interactor is exactly the same as interactor i;
    virtual bool isSame(coInteractor *i) const = 0;

    /// execute the Module
    virtual void executeModule() = 0;

    /// copy the Module to same host
    virtual void copyModule() = 0;

    /// copy the Module to same host and execute the copied one
    virtual void copyModuleExec() = 0;

    /// delete the Module
    virtual void deleteModule() = 0;

    // --- all getParameter Functions
    //       - return -1 on fail (illegal type requested), 0 if ok
    //       - only work for coFeedback created parameter messages
    //       - do not change the value fields if type incorrect

    virtual int getBooleanParam(const std::string &paraName, int &value) const = 0;
    virtual int getIntScalarParam(const std::string &paraName, int &value) const = 0;
    virtual int getFloatScalarParam(const std::string &paraName, float &value) const = 0;
    virtual int getIntSliderParam(const std::string &paraName, int &min, int &max, int &val) const = 0;
    virtual int getFloatSliderParam(const std::string &paraName, float &min, float &max, float &val) const = 0;
    virtual int getIntVectorParam(const std::string &paraName, int &numElem, int *&val) const = 0;
    virtual int getFloatVectorParam(const std::string &paraName, int &numElem, float *&val) const = 0;
    virtual int getStringParam(const std::string &paraName, const char *&val) const = 0;
    virtual int getChoiceParam(const std::string &paraName, int &num, char **&labels, int &active) const = 0;
    virtual int getFileBrowserParam(const std::string &paraName, char *&val) const = 0;

    // --- set-Functions:

    /// set Boolean Parameter
    virtual void setBooleanParam(const char *name, int val) = 0;

    /// set float scalar parameter
    virtual void setScalarParam(const char *name, float val) = 0;

    /// set int scalar parameter
    virtual void setScalarParam(const char *name, int val) = 0;

    /// set float slider parameter
    virtual void setSliderParam(const char *name, float min, float max, float value) = 0;

    /// set int slider parameter
    virtual void setSliderParam(const char *name, int min, int max, int value) = 0;

    /// set float Vector Param
    virtual void setVectorParam(const char *name, int numElem, float *field) = 0;
    virtual void setVectorParam(const char *name, float u, float v, float w) = 0;

    /// set int vector parameter
    virtual void setVectorParam(const char *name, int numElem, int *field) = 0;
    virtual void setVectorParam(const char *name, int u, int v, int w) = 0;

    /// set string parameter
    virtual void setStringParam(const char *name, const char *val) = 0;

    /// set choice parameter, pos starts with 1
    virtual void setChoiceParam(const char *name, int num, const char *const *list, int pos) = 0;

    /// set browser parameter
    virtual void setFileBrowserParam(const char *name, const char *val) = 0;

    // name of the covise data object which has feedback attributes
    virtual const char *getObjName() = 0;

    // the covise data object which has feedback attributes
    virtual RenderObject *getObject() = 0;

    // get the name of the module which created the data object
    virtual const char *getPluginName() = 0;

    // get the name of the module which created the data object
    virtual const char *getModuleName() = 0;

    // get the instance number of the module which created the data object
    virtual int getModuleInstance() = 0;

    // get the hostname of the module which created the data object
    virtual const char *getModuleHost() = 0;

    // -- The following functions only works for coFeedback attributes
    /// Get the number of Parameters
    virtual int getNumParam() const = 0;

    /// Get the number of User Strings
    virtual int getNumUser() const = 0;

    // get a User-supplied string
    virtual const char *getString(unsigned int i) const = 0;

private:
    int d_refCount;
};

class COVEREXPORT InteractorReference: public osg::Referenced
{
public:
    InteractorReference(opencover::coInteractor *inter)
    : m_interactor(inter)
    {
        if (m_interactor)
            m_interactor->incRefCount();
    }

    ~InteractorReference()
    {
        if (m_interactor)
            m_interactor->decRefCount();
    }

    opencover::coInteractor *interactor() const
    {
        return m_interactor;
    }

private:
    opencover::coInteractor *m_interactor = nullptr;
};

}
#endif
