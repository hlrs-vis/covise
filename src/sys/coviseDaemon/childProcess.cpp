/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "childProcess.h"
#ifndef _WIN32
#ifndef __APPLE__
#include <wait.h>
#endif
#include <unistd.h>
#include <errno.h>
#else
#include <tchar.h>
#include <stdio.h>
#include <strsafe.h>
#include <io.h>
#include <fcntl.h>
#endif // !_WIN32
#include <array>
#include <iostream>
#include <cstdio>
#include <sstream>
constexpr int BUFSIZE = 2048;

std::vector<const char *> stringToCharVec(const std::vector<std::string> &v)
{
	std::vector<const char *> argV{v.size() + 1};
	std::transform(v.begin(), v.end(), argV.begin(), [](const std::string &s)
				   { return s.c_str(); });
	argV.back() = nullptr;
	return argV;
}

#ifndef WIN32
void SigChildHandler::sigHandler(int sigNo) //  catch SIGTERM
{
	if (sigNo == SIGCHLD)
	{
		int pid = waitpid(-1, NULL, WNOHANG);
		std::cerr << "child with pid " << pid << " terminated" << std::endl;
		emit childDied(pid);
	}
}

const char *SigChildHandler::sigHandlerName() { return "sigChildHandler"; }

SigChildHandler childHandler;

#else

ProcessThread::ProcessThread(const std::vector<std::string> &args, QObject *parent) : m_args(args), QThread(parent)
{
}

void ProcessThread::run()
{
	//create pipe
	// Set the bInheritHandle flag so pipe handles are inherited.

	SECURITY_ATTRIBUTES saAttr;
	saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
	saAttr.bInheritHandle = TRUE;
	saAttr.lpSecurityDescriptor = NULL;

	// Create a pipe for the child process's STDOUT.
	if (!CreatePipe(&readHandle, &writeHandle, &saAttr, 0))
		return;

	//Ensure the read handle to the pipe for STDOUT is not inherited.
	if (!SetHandleInformation(readHandle, HANDLE_FLAG_INHERIT, 0))
		return;

	STARTUPINFO si;
	PROCESS_INFORMATION pi;

	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	//assign the write handle to the childs STDERR and STDOUT
	si.hStdError = writeHandle;
	si.hStdOutput = writeHandle;
	si.dwFlags |= STARTF_USESTDHANDLES;
	ZeroMemory(&pi, sizeof(pi));

	// Start the child process.
	std::stringstream ss;
	for (const auto &s : m_args)
		ss << s << " ";

	if (!CreateProcess(NULL,								 // No module name (use command line)
					   const_cast<char *>(ss.str().c_str()), // Command line
					   NULL,								 // Process handle not inheritable
					   NULL,								 // Thread handle not inheritable
					   TRUE,								 // Set handle inheritance to TRUE
					   0,									 // No creation flags
					   NULL,								 // Use parent's environment block
					   NULL,								 // Use parent's starting directory
					   &si,									 // Pointer to STARTUPINFO structure
					   &pi)									 // Pointer to PROCESS_INFORMATION structure
	)
	{
		printf("CreateProcess failed (%d).\n", GetLastError());

		return;
	}
	else
	{
		// Close handles to the stdin and stdout pipes no longer needed by the child process.
		// If they are not explicitly closed, there is no way to recognize that the child process has ended.
		CloseHandle(writeHandle);
	}
	ReadFromPipe();
	// Wait until child process exits.
	std::array<HANDLE, 2> handles{pi.hProcess, terminateHandle};

	WaitForMultipleObjects(2, handles.data(), FALSE, INFINITE);

	// Close process and thread handles.
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);
	CloseHandle(readHandle);
	emit died();
}

void ProcessThread::ReadFromPipe()
{
	DWORD numRead;
	CHAR buff[BUFSIZE];
	bool success = false;
	while (!m_terminate && (success = ReadFile(readHandle, buff, BUFSIZE, &numRead, NULL)))
	{
		std::string s(buff, numRead);
		emit output(QString{s.c_str()});
	}
}

void ProcessThread::terminate()
{
	m_terminate = true;
	SetEvent(terminateHandle);
	wait();
}

void ChildProcess::createWindowsProcess(const std::vector<std::string> &args)
{
	static int id = 0;
	++id;
	m_pid = id;
	auto p = new ProcessThread{args, this};
	connect(p, &ProcessThread::died, this, &ChildProcess::died);
	connect(p, &ProcessThread::output, this, &ChildProcess::output);
	connect(p, &ProcessThread::finished, p, &ProcessThread::deleteLater);
	connect(this, &ChildProcess::destructor, p, &ProcessThread::terminate);
	p->start();
}
#endif

ChildProcess::ChildProcess(const char *path, const std::vector<std::string> &args)
{
	std::vector<std::string> argS{args.size() + 1};
	argS[0] = path;
	for (size_t i = 0; i < args.size(); i++)
	{
		argS[i + 1] = args[i].c_str();
	}

#ifdef _WIN32
	createWindowsProcess(argS);
#else
	auto argV = stringToCharVec(argS);
	int pipefd[2];
	if (pipe(pipefd) == -1)
	{
		std::cerr << "coviseDaemon: could not create pipe for executing " << path << ": " << strerror(errno) << std::endl;
		return;
	}
	m_pid = fork();
	if (m_pid == -1)
	{
		std::cerr << "coviseDaemon: fork() for executing " << path << " failed: " << strerror(errno) << std::endl;
		close(pipefd[0]);
		close(pipefd[1]);
		return;
	}
	else if (m_pid == 0)
	{
		close(pipefd[0]); // close reading end in the child

		dup2(pipefd[1], 1); // send stdout to the pipe
		dup2(pipefd[1], 2); // send stderr to the pipe

		close(pipefd[1]); // this descriptor is no longer needed

		if (execvp(path, const_cast<char *const *>(argV.data())) == -1)
		{
			std::cerr << "coviseDaemon: failed to exec " << path << ": " << strerror(errno) << std::endl;
			exit(1);
		}
	}
	else
	{
		// Needed to prevent zombies
		// if childs terminate
		//signal(SIGCHLD, SIG_IGN);
		covise::coSignal::addSignal(SIGCHLD, childHandler);
		connect(&childHandler, &SigChildHandler::childDied, this, [this](int pid)
				{
					if (pid == m_pid)
						emit died();
				});
		char buffer[BUFSIZE];
		close(pipefd[1]); // close the write end of the pipe in the parent
		m_outputNotifier.reset(new QSocketNotifier{pipefd[0], QSocketNotifier::Type::Read});
		connect(m_outputNotifier.get(), &QSocketNotifier::activated, this, [this, pipefd]()
				{
					char buffer[BUFSIZE];
					int num = 0;
					if ((num = ::read(pipefd[0], buffer, sizeof(buffer))) != 0)
					{
						std::string msg(buffer, num);
						emit output(QString(msg.c_str()));
					}
				});
		connect(this, &ChildProcess::destructor, this, [pipefd]()
				{ close(pipefd[0]); });
	}
#endif
}

ChildProcess::~ChildProcess()
{
	emit destructor(); //terminate the child thread
}

bool ChildProcess::operator<(const ChildProcess &other) const
{
	return m_pid < other.m_pid;
}

bool ChildProcess::operator==(const ChildProcess &other) const
{
	return m_pid == other.m_pid;
}
