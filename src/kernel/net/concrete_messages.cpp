#include "concrete_messages.h"
#include "message.h"
#include "message_sender_interface.h"
#include "message_types.h"
#include "tokenbuffer.h"
#include "tokenbuffer_util.h"
#include "tokenbuffer_serializer.h"

#include <cassert>
#include <iostream>
#include <algorithm>
#include <functional>
namespace covise {

TokenBuffer& operator<<(TokenBuffer& tb, ExecFlag flag) {
	tb << static_cast<int>(flag);
	return tb;
}

std::ostream& operator<<(std::ostream& os, ExecFlag flag) {
	os << static_cast<int>(flag);
	return os;
}

TokenBuffer& operator>>(TokenBuffer& tb, ExecFlag& flag) {
	int i;
	tb >> i;
	flag = static_cast<ExecFlag>(i);
	return tb;
}

IMPL_MESSAGE_CLASS(CRB_EXEC,
	ExecFlag, flag,
	char*, name,
	int, port,
	char*, localIp,
	int, moduleCount,
	char*, moduleId,
	char*, moduleIp,
	char*, moduleHostName,
	char*, displayIp,
	char*, category,
	int, vrbClientIdOfController,
	std::vector<std::string>, params)


std::string charToString(const char* c) {
	if (c && !c[0] == '\0')
	{
		return std::string{ c };
	}
	return std::string{ "INVALID" };
}

const char* adoptedChar(const char* c) {
	return (c && strcmp(c, "INVALID")) ? c : nullptr;
}

std::vector<std::string> getCmdArgs(const CRB_EXEC& exec) {

	size_t l = 10;
	l += exec.params.size();
	std::vector<std::string> args(l);
	size_t pos = 0;
	args[pos++] = charToString(exec.name);
	args[pos++] = std::to_string(exec.port);
	args[pos++] = charToString(exec.localIp);
	args[pos++] = std::to_string(exec.moduleCount);
	args[pos++] = charToString(exec.moduleId);
	args[pos++] = charToString(exec.moduleIp);
	args[pos++] = charToString(exec.moduleHostName);
	args[pos++] = charToString(exec.displayIp);
	args[pos++] = std::to_string(exec.vrbClientIdOfController);
	args[pos++] = std::to_string(exec.params.size());
	for (auto& arg : exec.params)
	{
		args[pos++] = arg;
	}
	return args;
}

void invalidArgsError(int argC, int expected, std::function<bool(int, int)> compare, char* argV[]) {
	if (!compare(argC, expected))
	{
		std::cerr << "Application Module with inappropriate arguments called: "
		 << argC << ", " << expected << " expected" << std::endl;
		for (int i = 0; i < argC; ++i)
		{
			std::cerr << i << ": " << argV[i] << std::endl;
		}
		assert(false);
		exit(1);
	}
}

CRB_EXEC getExecFromCmdArgs(int argC, char* argV[]) {


	invalidArgsError(argC, 10, 
					[](int a, int b){ return a >= b; },
	 				argV);
	int numExtraArgs = std::stoi(argV[9]);
	invalidArgsError(argC, numExtraArgs + 10, 
					[](int a, int b){ return a == b; },
	 				argV);
	std::vector<std::string> extraArgs(numExtraArgs);
	for (size_t i = 0; i < numExtraArgs; i++)
	{
		extraArgs[i] = argV[i + 10];
	}
	CRB_EXEC exec(ExecFlag::Normal,
		adoptedChar(argV[0]), //name
		atoi(argV[1]), //port
		adoptedChar(argV[2]), //localIp
		atoi(argV[3]), //moduleCount
		adoptedChar(argV[4]), //moduleId
		adoptedChar(argV[5]), //moduleIp
		adoptedChar(argV[6]), //moduleHostName
		adoptedChar(argV[7]), //displayIp
		nullptr, //category
		atoi(argV[8]), //vrbClientIdOfController
		extraArgs); //params
	return exec;
}

std::vector<const char*> cmdArgsToCharVec(const std::vector<std::string>& args) {
	std::vector<const char*> v(args.size() + 1);
	std::transform(args.begin(), args.end(), v.begin(), [](const std::string& s) {return s.c_str(); });
	v[args.size()] = nullptr;
	return v;
}

} //covise



