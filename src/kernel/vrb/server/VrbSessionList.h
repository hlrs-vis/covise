#ifndef VRB_SESSION_LIST_H
#define VRB_SESSION_LIST_H

#include "VrbServerRegistry.h"
#include <vector>

namespace covise {
class TokenBuffer;
}
namespace vrb {
class VRBSClient;
class VrbSessionList {
public:
	typedef std::vector<VrbServerRegistry> ValueType;
	typedef ValueType::iterator Iter;
	typedef ValueType::const_iterator Const_Iter;
	VrbSessionList();
	VrbSessionList(const VrbSessionList&) = delete;
	VrbSessionList(VrbSessionList&&) = delete;
	VrbSessionList operator=(const VrbSessionList&) = delete;
	VrbSessionList operator=(VrbSessionList&&) = delete;


	VrbServerRegistry& operator[](const SessionID& id);
	const VrbServerRegistry& operator[](const SessionID& id) const;

	VrbServerRegistry& forceCreateSession(const SessionID& id);
	void unobserveFromAll(int senderID, const std::string& className, const std::string& varName);
	void unobserveSession(int observer, const SessionID& id);

	bool serializeSessions(covise::TokenBuffer& tb);
	void disconectClientFromSessions(int clientID);
	covise::TokenBuffer serializeSession(const SessionID& id) const;
	const VrbServerRegistry &deserializeSession(covise::TokenBuffer& tb, const SessionID& id);
	void setMaster(const SessionID& sid);
private:
	const SessionID vrbSession = SessionID(0, std::string(), false);
	ValueType m_sessions;
	Const_Iter find(const SessionID& id) const;
	Iter find(const SessionID& id);
	Const_Iter begin() const;
	Iter begin();
	Const_Iter end() const;
	Iter end();
	std::vector<VRBSClient*> getParticipants(const SessionID& id) const;
	std::string getCurrentTime() const;
};

}

#endif