#include "ClientRegistryClass.h"
#include "ClientRegistryVariable.h"
#include "VrbClientRegistry.h"

using namespace covise;
using namespace vrb;


void clientRegVar::notifyLocalObserver()
{
    if (_observer)
    {
        _observer->update(this);
    }
}

void clientRegVar::subscribe(regVarObserver * ob, const SessionID &sessionID)
{
    _observer = ob;
    TokenBuffer tb;
    // compose message
    tb << sessionID;
    tb << m_class->getID();
    tb << m_class->name();
    tb << m_name;
    sendValue(tb);
    // inform vrb about creation
    dynamic_cast<clientRegClass *>(m_class)->sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE);
}