#include "VrbSessionList.h"
#include "VrbClientList.h"
#include <boost/chrono/time_point.hpp>
using namespace vrb;

vrb::VrbSessionList::VrbSessionList()
{
	operator[](vrbSession);
}

VrbServerRegistry& vrb::VrbSessionList::operator[](const SessionID& id)
{
	auto it = find(id);
	if (it == end())
	{
		it = m_sessions.emplace(end(), VrbServerRegistry{ id });
	}
	return *it;
}

const VrbServerRegistry& vrb::VrbSessionList::operator[](const SessionID& id) const
{
	return const_cast<VrbSessionList*>(this)->operator[](id);
}

VrbServerRegistry& vrb::VrbSessionList::forceCreateSession(const SessionID& id)
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
	auto it = m_sessions.emplace(end(), VrbServerRegistry{ newID });
	it->setOwner(id.owner());
	return *it;
}

VrbServerRegistry& vrb::VrbSessionList::createSessionIfnotExists(const vrb::SessionID& sessionID, VRBSClient* cl)
{
	auto ses = find(sessionID);
	if (ses == end())
	{
		if (cl && !sessionID.isPrivate())
		{
			cl->setSession(sessionID);
		}
		ses = m_sessions.emplace(end(), VrbServerRegistry{ sessionID });
	}
	return *ses;
}

void vrb::VrbSessionList::unobserveFromAll(int senderID, const std::string& className, const std::string& varName)
{
	for (auto& reg : m_sessions)
		reg.unObserveVar(senderID, className, varName);
}

void vrb::VrbSessionList::unobserveSession(int observer, const SessionID& id)
{
	auto s = find(id);
	if (s != end())
	{
		s->unObserve(observer);
	}
}

int vrb::VrbSessionList::getSessionOwner(const SessionID& id) const
{
	auto s = find(id);
	if (s != end())
	{
		return s->getOwner();
	}
	return id.owner();
}

bool vrb::VrbSessionList::serializeSessions(covise::TokenBuffer& tb)
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

void vrb::VrbSessionList::disconectClientFromSessions(int clientID)
{
	auto session = begin();
	while (session != end())
	{
		const auto& sid = session->sessionID();
		if (sid.owner() == clientID)
		{
			if (sid.isPrivate())
			{
				session = m_sessions.erase(session);
			}
			else
			{
				auto newOwner = clients.getNextInGroup(sid);

				if (newOwner)
				{
					sid.setOwner(newOwner->getID());
				}
				else
				{
					session = m_sessions.erase(session); //detele session if there are no more clients in it
				}
			}
		}
		else
		{
			++session;
		}


	}
}

covise::TokenBuffer vrb::VrbSessionList::serializeSession(const SessionID& id) const
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
	outData << participants.size();
	for (const auto& cl : participants)
	{
		outData << cl->getUserName();
	}

	for (const auto& cl : participants) {
		operator[](cl->getPrivateSession()).serialize(outData);
	}
	//write shared session after private sessions to ensure that sessions get merged correctly in case of currenSession.isPrivate()
	sharedSession->serialize(outData);
	return outData;
}

const VrbServerRegistry &vrb::VrbSessionList::deserializeSession(covise::TokenBuffer& tb, const SessionID& id)
{
	std::string time;
	tb >> time;
	size_t numberOfParticipants;
	tb >> numberOfParticipants;
	std::vector<std::string> participantNames(numberOfParticipants);
	for (size_t i = 0; i < numberOfParticipants; i++)
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



VrbSessionList::Const_Iter vrb::VrbSessionList::find(const SessionID& id) const
{
	return std::find_if(begin(), end(), [id](const ValueType::value_type& registry) {return registry.sessionID() == id; });
}

VrbSessionList::Iter vrb::VrbSessionList::find(const SessionID& id) 
{
	return std::find_if(begin(), end(), [id](const ValueType::value_type& registry) {return registry.sessionID() == id; });
}

VrbSessionList::Const_Iter vrb::VrbSessionList::begin() const
{
	return m_sessions.begin();
}

VrbSessionList::Iter vrb::VrbSessionList::begin() 
{
	return m_sessions.begin();
}

VrbSessionList::Const_Iter vrb::VrbSessionList::end() const
{
	return m_sessions.end();
}

VrbSessionList::Iter vrb::VrbSessionList::end() 
{
	return m_sessions.end();
}

std::vector<VRBSClient*> vrb::VrbSessionList::getParticipants(const SessionID& id) const
{
	std::vector<VRBSClient*> participants;
	for (size_t i = 0; i < clients.numberOfClients(); i++)
	{
		if (clients.getNthClient(i)->getSession() == id)
		{
			participants.push_back(clients.getNthClient(i));
		}
	}
	return participants;
}

std::string vrb::VrbSessionList::getCurrentTime() const
{
	time_t rawtime;
	time(&rawtime);
	struct tm* timeinfo = localtime(&rawtime);

	char buffer[80];
	strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", timeinfo);
	std::string str(buffer);
	return str;
}
