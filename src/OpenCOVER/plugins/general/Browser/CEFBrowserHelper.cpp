

#ifdef WIN32
#include <windows.h>
#endif
#include <type_traits>
#include <iostream>
#include "include/cef_app.h"
#include "include/base/cef_logging.h"
#include "include/cef_command_line.h"
#ifdef WIN32
// Entry point function for all processes.
int APIENTRY WinMain(HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPTSTR lpCmdLine,
    int nCmdShow) {
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // Enable High-DPI support on Windows 7 or newer.
    CefEnableHighDPISupport();

    void* sandbox_info = nullptr;
    //std::cerr << lpCmdLine << "CEFBrowserHelper" << std::endl;

#if defined(CEF_USE_SANDBOX)
    // Manage the life span of the sandbox information object. This is necessary
    // for sandbox support on Windows. See cef_sandbox_win.h for complete details.
    CefScopedSandboxInfo scoped_sandbox;
    sandbox_info = scoped_sandbox.sandbox_info();
#endif

    // Provide CEF with command-line arguments.
    CefMainArgs main_args(hInstance);

    // CEF applications have multiple sub-processes (render, plugin, GPU, etc)
    // that share the same executable. This function checks the command-line and,
    // if this is a sub-process, executes the appropriate logic.
    int exit_code = CefExecuteProcess(main_args, nullptr, sandbox_info);
    if (exit_code >= 0) {
        // The sub-process has completed so return here.
        return exit_code;
    }
    return -1;
}
#else
// Entry point function for all processes.
int main(int argc, char* argv[]) {
    // Provide CEF with command-line arguments.
    CefMainArgs main_args(argc, argv);

    // CEF applications have multiple sub-processes (render, plugin, GPU, etc)
    // that share the same executable. This function checks the command-line and,
    // if this is a sub-process, executes the appropriate logic.
    int exit_code = CefExecuteProcess(main_args, nullptr, nullptr);
    if (exit_code >= 0) {
        // The sub-process has completed so return here.
        return exit_code;
    }
}
#endif
