solution "ASIO"

    configurations { "Debug", "Release" }

    platforms { "x64", "x32" }

    location    ("build/" .. _ACTION)
    objdir      ("build/" .. _ACTION .. "/obj")
    targetdir   ("build/" .. _ACTION .. "/bin")

--    defines {
--        "VV_S11N_TEXT_ARCHIVE"
--    }

    configuration { "Debug" }

        defines { "_DEBUG" }
        flags { "ExtraWarnings", "Symbols" }

    configuration { "Release" }

        defines { "NDEBUG" }
        flags { "ExtraWarnings", "Optimize" }

    configuration { "gmake" }

        buildoptions {
--            "-std=c++03",
            "-Wall",
            "-Wextra",
            "-pedantic",
        }

    configuration { "windows" }

        flags { "Unicode" }

----------------------------------------------------------------------------------------------------
project "virvo"

    kind "StaticLib"

    language "C++"
    
    includedirs {
        "../../virvo",
        "../../virvo/private",
    }

    files {
        "../../virvo/private/vvclient.cpp",
        "../../virvo/private/vvmessage.cpp",
        "../../virvo/private/vvserver.cpp",
    }

    configuration { "gmake" }
        buildoptions { "-std=c++03" }

----------------------------------------------------------------------------------------------------
project "client"

    kind "ConsoleApp"

    language "C++"
    
    includedirs {
        "../../virvo",
        "../../virvo/private",
    }

    files {
        "client.cpp",
    }

    links { "virvo", "boost_system", "boost_serialization" }

    configuration { "gmake" }
        buildoptions { "-std=c++11" } -- std::thread

    configuration { "windows" }
        links { "ws2_32", "mswsock" }

----------------------------------------------------------------------------------------------------
project "server"

    kind "ConsoleApp"

    language "C++"
    
    includedirs {
        "../../virvo",
        "../../virvo/private",
    }

    files {
        "server.cpp",
    }

    links { "virvo", "boost_system", "boost_serialization" }

    configuration { "gmake" }
        buildoptions { "-std=c++03" }

    configuration { "windows" }
        links { "ws2_32", "mswsock" }

----------------------------------------------------------------------------------------------------
if _ACTION == "clean" then
    os.rmdir("build")
end
