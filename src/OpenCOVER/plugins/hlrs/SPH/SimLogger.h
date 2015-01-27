/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// #ifndef __SimLogger_h__
// #define __SimLogger_h__
//
// #include <map>
// #include <iostream>
// #include <fstream>
// #include <iomanip>
// #include <vector>
// #include <string>
//
// using namespace std;
//
// namespace SimLib {
//
//
// 	enum SimLogLevel
// 	{
// 		Info,
// 		Warn,
// 		Error,
// 		Debug,
// 	};
//
// 	class SimLogger
// 	{
// 	private:
// 		char tmp[8*1024];
// 	public:
// 		SimSettings()
// 		{
// 		}
// 		~SimSettings()
// 		{
// 		}
//
// 		void Log(SimLogLevel logLevel, std::string string)
// 		{
// 			//TODO: loglevel etc
// 			cout << string;
// 		}
// 		void Log(std::string string)
// 		{
// 			Log(Info, string);
// 		}
//
// 		int Log(SimLogLevel logLevel, char* fmt, ...){
//
// 			int retval=0;
// 			va_list ap;
//
// 			va_start(ap, fmt); /* Initialize the va_list */
//
// 			retval = vsprintf(fmt, ap); /* Call vprintf */
//
// 			va_end(ap); /* Cleanup the va_list */
//
// 			Log(logLevel, fmt);
//
// 			return retval;
//
// 		}
// 	};
//
// } // namespace SimLib
//
// #endif