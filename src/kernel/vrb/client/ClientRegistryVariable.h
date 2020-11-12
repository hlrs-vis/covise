#ifndef VRB_CLIENT_REGISTRY_VARIABLE_H
#define VRB_CLIENT_REGISTRY_VARIABLE_H

#include <vrb/RegistryVariable.h>
#include <net/tokenbuffer.h>
#include <net/tokenbuffer_serializer.h>

#include <util/coExport.h>

#include <string>
#include <map>

namespace vrb
{

class VRBCLIENTEXPORT clientRegVar : public regVar
{
private:
    regVarObserver *_observer = nullptr;
    int lastEditor = -1;
public:
    using regVar::regVar;
    ///returns the clent side observer
    regVarObserver * getLocalObserver()
    {
        return _observer;
    }
    void notifyLocalObserver();
    void subscribe(regVarObserver *ob, const SessionID &sessionID);

    //void attach(regVarObserver *ob)
    //{
    //    _observer = ob;
    //}
    int getLastEditor()
    {
        return lastEditor;
    }
    void setLastEditor(int lastEditor)
    {
        this->lastEditor = lastEditor;
    }
};


} // namespace vrb

#endif