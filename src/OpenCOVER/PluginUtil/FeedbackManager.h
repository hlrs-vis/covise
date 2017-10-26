#ifndef COVER_FEEDBACKMANAGER_H
#define COVER_FEEDBACKMANAGER_H

#include <util/coExport.h>

#include <map>

namespace opencover
{

class coInteractor;
class ModuleInteraction;

class PLUGIN_UTILEXPORT FeedbackManager
{
public:
    static FeedbackManager *instance();
    ~FeedbackManager();

    void registerFeedback(ModuleInteraction *feedback, coInteractor *inter=nullptr);
    void unregisterFeedback(ModuleInteraction *feedback);

    ModuleInteraction *findFeedback(coInteractor *inter) const;

private:
    std::map<coInteractor *, ModuleInteraction *> m_moduleFeedback;

    static FeedbackManager *s_instance;
    FeedbackManager();
};

}
#endif
