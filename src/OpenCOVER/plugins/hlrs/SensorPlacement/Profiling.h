#pragma once

#include <string>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <thread>
#include <mutex>
/*
 Usage: include this header file somewhere in your code (eg. precompiled header), and then use like:

 Instrumentor::Get().BeginSession("Session Name");        // Begin session 
 {
     InstrumentationTimer timer("Profiled Scope Name");   // Place code like this in scopes you'd like to include in profiling Code
 }
 Instrumentor::Get().EndSession();                        // End Session
 */

#define SP_PROFILE 0

namespace SP
{
    struct ProfileResult
    {
        const std::string name;
        long long start, end;
        uint32_t threadID;
    };

    class Instrumentor
    {
        std::string     m_sessionName   = "None";
        std::ofstream   m_outputStream;
        int             m_profileCount  = 0;
        std::mutex      m_lock;
        bool            m_activeSession = false;

        Instrumentor() { }

    public:

        static Instrumentor& GetInstance()
        {
            static Instrumentor instance;
            return instance;
        }

        ~Instrumentor()
        {
            endSession();
        }

        void beginSession(const std::string& name, const std::string& filepath = "results.json")
        {
            if (m_activeSession) { endSession(); }
            m_activeSession = true;
            m_outputStream.open(filepath);
            writeHeader();
            m_sessionName = name;
        }

        void endSession()
        {
            if (!m_activeSession) { return; }
            m_activeSession = false;
            writeFooter();
            m_outputStream.close();
            m_profileCount = 0;
        }

        void writeProfile(const ProfileResult& result)
        {
            std::lock_guard<std::mutex> lock(m_lock);

            if (m_profileCount++ > 0) { m_outputStream << ","; }

            std::string name = result.name;
            std::replace(name.begin(), name.end(), '"', '\'');

            m_outputStream << "{";
            m_outputStream << "\"cat\":\"function\",";
            m_outputStream << "\"dur\":" << (result.end - result.start) << ',';
            m_outputStream << "\"name\":\"" << name << "\",";
            m_outputStream << "\"ph\":\"X\",";
            m_outputStream << "\"pid\":0,";
            m_outputStream << "\"tid\":" << result.threadID << ",";
            m_outputStream << "\"ts\":" << result.start;
            m_outputStream << "}";
        }

        void writeHeader()
        {
            m_outputStream << "{\"otherData\": {},\"traceEvents\":[";
        }

        void writeFooter()
        {
            m_outputStream << "]}";
        }
    };

    class InstrumentationTimer
    {
        ProfileResult m_result;

        std::chrono::time_point<std::chrono::high_resolution_clock> m_startTimepoint;
        bool m_stopped;

    public:

        InstrumentationTimer(const std::string & name)
            : m_result({ name, 0, 0, 0 })
            , m_stopped(false)
        {
            m_startTimepoint = std::chrono::high_resolution_clock::now();
        }

        ~InstrumentationTimer()
        {
            if (!m_stopped) { stop(); }
        }

        void stop()
        {
            auto endTimepoint = std::chrono::high_resolution_clock::now();

            m_result.start = std::chrono::time_point_cast<std::chrono::microseconds>(m_startTimepoint).time_since_epoch().count();
            m_result.end   = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();
            m_result.threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
            Instrumentor::GetInstance().writeProfile(m_result);

            m_stopped = true;
        }
    };
}

#if SP_PROFILE
    // Resolve which function signature macro will be used. Note that this only
	// is resolved when the (pre)compiler starts, so the syntax highlighting
	// could mark the wrong one in your editor!
	#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)
		#define SP_FUNC_SIG __PRETTY_FUNCTION__
	#elif defined(__DMC__) && (__DMC__ >= 0x810)
		#define SP_FUNC_SIG __PRETTY_FUNCTION__
	#elif defined(__FUNCSIG__)
		#define SP_FUNC_SIG __FUNCSIG__
	#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
		#define SP_FUNC_SIG __FUNCTION__
	#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
		#define SP_FUNC_SIG __FUNC__
	#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
		#define SP_FUNC_SIG __func__
	#elif defined(__cplusplus) && (__cplusplus >= 201103)
		#define SP_FUNC_SIG __func__
	#else
		#define SP_FUNC_SIG "SP_FUNC_SIG unknown!"
	#endif

    #define SP_PROFILE_BEGIN_SESSION(name,filepath) ::SP::Instrumentor::GetInstance().beginSession(name,filepath)
    #define SP_PROFILE_END_SESSION()    ::SP::Instrumentor::GetInstance().endSession()
    #define SP_PROFILE_SCOPE(name)   ::SP::InstrumentationTimer timer##__LINE__(name);
    #define SP_PROFILE_FUNCTION()   SP_PROFILE_SCOPE(SP_FUNC_SIG) 
#else
    #define SP_PROFILE_BEGIN_SESSION(name,filepath)
    #define SP_PROFILE_END_SESSION()
    #define SP_PROFILE_SCOPE(name)
    #define SP_PROFILE_FUNCTION()
#endif
