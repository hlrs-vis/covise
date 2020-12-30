#include "CRB_EXEC.h"
#include <net/concrete_messages.h>
#include <cassert>
#include <cstring>
namespace test {
void test_crbExec() {
	std::vector<std::string> params;
	params.push_back("test8");
	params.push_back("test9");

	covise::CRB_EXEC crbExec1{ covise::ExecFlag::Memcheck, "test1", 31000, "test2", 5, "test3", "test4", "test5", nullptr, nullptr, 13, vrb::VrbCredentials{"test", 83211, 934782}, params };

	//std::cerr << crbExec1 << std::endl << std::endl;
	auto a = covise::getCmdArgs(crbExec1);
	auto args = covise::cmdArgsToCharVec(a);
	covise::CRB_EXEC crbExec2 = covise::getExecFromCmdArgs(a.size(), const_cast<char**>(args.data()));
	//std::cerr << crbExec2 << std::endl << std::endl;
	assert(!strcmp(crbExec1.name, crbExec2.name));
	assert(!strcmp(crbExec1.localIp, crbExec2.localIp));
	assert(!strcmp(crbExec1.moduleId, crbExec2.moduleId));
	assert(!strcmp(crbExec1.moduleIp, crbExec2.moduleIp));
	assert(!strcmp(crbExec1.moduleHostName, crbExec2.moduleHostName));
	assert(!crbExec2.displayIp);
	assert(!crbExec2.category);
	assert(crbExec1.params[0] == crbExec2.params[0]);
	assert(crbExec1.params[1] == crbExec2.params[1]);
	assert(crbExec1.vrbCredentials.ipAddress == crbExec2.vrbCredentials.ipAddress);
	assert(crbExec1.vrbCredentials.tcpPort == crbExec2.vrbCredentials.tcpPort);
	assert(crbExec1.vrbCredentials.udpPort == crbExec2.vrbCredentials.udpPort);
}


}