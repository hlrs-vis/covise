#include "VrbSessionList.h"
#include "VrbClientList.h"

#include <boost/chrono/time_point.hpp>

#include <algorithm>
#include <chrono>
using namespace vrb;

VrbSessionList::VrbSessionList()
{
	operator[](vrbSession);
}


VrbServerRegistry& VrbSessionList::operator[](const SessionID& id)
{
	auto it = find(id);
	VrbServerRegistry t{ id };
	if (it == end())
	{
		it = m_sessions.emplace(end(), VrbServerRegistry{ id });
	}
	return *it;
}

const VrbServerRegistry& VrbSessionList::operator[](const SessionID& id) const
{
	return const_cast<VrbSessionList*>(this)->operator[](id);
}

VrbServerRegistry& VrbSessionList::forceCreateSession(const SessionID& id)
{
	int genericName = 0;
	std::string name = id.name();
	auto newID = id;
	if (name == std::string()) //unspecific name or already existing session -> create generic name here
	{
		++genericName;
		newID.setName("1");
	}
	while (find(newID) != end())
	{
		++genericName;
		newID.setName(name + std::to_string(genericName));

	}
	return *m_sessions.emplace(end(), VrbServerRegistry{ newID });
}

void VrbSessionList::unobserveFromAll(int senderID, const std::string& className, const std::string& varName)
{
	for (auto& reg : m_sessions)
		reg.unObserveVar(senderID, className, varName);
}

void VrbSessionList::unobserveSession(int observer, const SessionID& id)
{
	auto s = find(id);
	if (s != end())
	{
		s->unObserve(observer);
	}
}

bool VrbSessionList::serializeSessions(covise::TokenBuffer& tb)
{
	auto ph = tb.addPlaceHolder<int>();
	int size = 0;
	for (const auto &session: m_sessions)
	{
		if (session.sessionID() != vrbSession)
		{
			tb << session.sessionID();
			++size;
		}
	}
	ph.replace(size);
	return size > 0;
}

void VrbSessionList::disconectClientFromSessions(int clientID)
{
	auto registry = begin();
	while (registry != end())
	{
		auto& sid = registry->sessionID();
		if (sid.owner() == clientID)
		{
			auto newOwner = clients.getNextInGroup(sid);
			if (!newOwner)
			{
				registry = m_sessions.erase(registry); //detele session if there are no more clients in it
			}
			else
			{
				sid.setOwner(newOwner->ID());
				sid.setMaster(newOwner->ID());
				++registry;
			}
		}
		else
		{
			++registry;
		}
	}
}

covise::TokenBuffer VrbSessionList::serializeSession(const SessionID& id) const
{
	auto participants = getParticipants(id);
	covise::TokenBuffer outData;
	auto sharedSession = find(id);
	if (sharedSession == end())
	{
		std::cerr << "failed to serialize session " << id << ": noo such session!" << std::endl;
		return outData;
	}
	outData << getCurrentTime();
	outData << (uint32_t)participants.size();
	for (const auto& cl : participants)
	{
		outData << cl->userInfo().userName;

	}

	for (const auto& cl : participants) {
		operator[](cl->getPrivateSession()).serialize(outData);
	}
	//write shared session after private sessions to ensure that sessions get merged correctly in case of currenSession.isPrivate()
	sharedSession->serialize(outData);
	return outData;
}

const VrbServerRegistry &VrbSessionList::deserializeSession(covise::TokenBuffer& tb, const SessionID& id)
{
	std::string time;
	tb >> time;
	uint32_t numberOfParticipants;
	tb >> numberOfParticipants;
	std::vector<std::string> participantNames(numberOfParticipants);
	for (uint32_t i = 0; i < numberOfParticipants; i++)
	{
		tb >> participantNames[i];
	}
	//read and assign private registries
	std::map<std::string, size_t> numParticipantDoubles;
	for (const auto par : participantNames)
	{
		numParticipantDoubles[par]++;
	}
	for (const auto& d : numParticipantDoubles)
	{
		const auto cls = clients.getClientsWithUserName(d.first);
		for (size_t i = 0; i < std::min(d.second, cls.size()); i++)
		{
			operator[](cls[i]->getPrivateSession()).deserialize(tb);
		}
	}
	//read shared session after private sessions to ensure that sessions get merged correctly in case of currenSession.isPrivate()
	auto &registry = operator[](id);
	registry.deserialize(tb);
	return registry;
}

void VrbSessionList::setMaster(const SessionID& sid) {
	auto reg = find(sid);
	reg->sessionID().setMaster(sid.master());
	reg->sessionID().setOwner(sid.owner());
}


VrbSessionList::Const_Iter VrbSessionList::find(const SessionID& id) const
{
	return std::find_if(begin(), end(), [id](const ValueType::value_type& registry) {return registry.sessionID() == id; });
}

VrbSessionList::Iter VrbSessionList::find(const SessionID& id) 
{
	return std::find_if(begin(), end(), [id](const ValueType::value_type& registry) {return registry.sessionID() == id; });
}

VrbSessionList::Const_Iter VrbSessionList::begin() const
{
	return m_sessions.begin();
}

VrbSessionList::Iter VrbSessionList::begin() 
{
	return m_sessions.begin();
}

VrbSessionList::Const_Iter VrbSessionList::end() const
{
	return m_sessions.end();
}

VrbSessionList::Iter VrbSessionList::end() 
{
	return m_sessions.end();
}

std::vector<VRBSClient*> VrbSessionList::getParticipants(const SessionID& id) const
{
	std::vector<VRBSClient*> participants;
	for (size_t i = 0; i < clients.numberOfClients(); i++)
	{
		if (clients.getNthClient(i)->sessionID() == id)
		{
			participants.push_back(clients.getNthClient(i));
		}
	}
	return participants;
}

std::string VrbSessionList::getCurrentTime() const
{
	time_t rawtime;
	time(&rawtime);
	struct tm* timeinfo = localtime(&rawtime);

	char buffer[80];
	strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", timeinfo);
	std::string str(buffer);
	return str;
}
