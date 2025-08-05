# The meaning of the variable ARCHSUFFIX
# ======================================
# 
# The variable ARCHSUFFIX specifies the system environment for which COVISE
# is compiled. It depends on the combination of
# - the system architecture (e.g. ia32, x86_64, mips4, ...)
# - the operating system (e.g. Windows XP SP2, Fedora Linux Core 3, Red Hat Enterprise Linux 3, ...)
# - the programming model (e.g. N32, O32, 64 bit, 32 bit, ...)
# - the version of your system libraries (e.g. glibc 2.3.2, 
# - the C++ ABI version (e.g. as used by GCC 2.95/2.96, GCC 3.0, GCC 3.1, GCC 3.2/3.3, GCC 3.4/4.0, ...)
# - the compiler optimization level
# - ...
# 
# You can append 'opt' to every base ARCHSUFFIX name (as listed below) in
# order to obtain one for optimized builds.  
# 
# 
# Adding a new ARCHSUFFIX
# =======================
# 
# You have to change or create at least the following:
# covise/scripts/covise-functions.sh
# covise/README-ARCHSUFFIX.txt (this file)
# covise/archive/extlibs-$BASEARCHSUFFIX-files.txt
# covise/.gitignore
# 
# 
# ARCHSUFFIXes currently used
# ===========================
# 
# ARCHSUFFIX    Meaning
# ----------------------------------------------------------------------

zebu            Visual Studio 2015 Update 3, 2017 or 2019 for x86_64

linux32         Generic x86 Linux
linux64         Generic x86_64 Linux
linuxarm        Generic aarch64 Linux
rhel8           Red Hat Enterprise Linux/CentOS 8 with updates on x86_64 (64 bit mode)
rhel9           Red Hat Enterprise Linux/CentOS/Rocky 9 with updates on x86_64 (64 bit mode)
jammy           Ubuntu 22.04 Jammy Jellyfish on x86_64 (64 bit mode)
noble		Ubuntu 24.04 Noble Numbat on x86_64 (64 bit mode)

darwin          Generic macOS
macos           macOS, x86_64 or arm64 with libc++

spack*          Installation of dependencies via spack (https://spack.io), possibly based on a spack environment


#
# outdated, unsupported ARCHSUFFIXes
# ==================================
#
# ARCHSUFFIX    Meaning
# ----------------------------------------------------------------------

win32           Windows 2000/XP with Visual Studio 2003 on ia32
vista           Windows XP/Vista with Visual Studio 2005 with SP1 on ia32
zackel          Windows XP/Vista with Visual Studio 2008 on ia32
amdwin64        Windows XP/Vista with Visual Studio 2005 with SP1 on x86_64
angus           Windows XP/Vista with Visual Studio 2008 on x86_64
yoroo           Visual Studio 2010 for ia32
berrenda        Visual Studio 2010 for x86_64
tamarau         Visual Studio 2012 for x86_64
mingw           Windows XP/Vista with gcc 4.4.0 on ia32
vcpkg           Visual Studio 2015 Update 3, 2017 or 2019 with 3rd party libraries from vcpkg

freebsd         Generic FreeBSD
cray64          Cray x86_64 Linux software rendering

linux           Red Hat 7.3 (with GCC 2.96) on ia32
gcc3            Red Hat 8 with glibc 2.3.2 on ia32
rhel3           Red Hat Enterprise Linux 3 with updates on x86_64 (64 bit mode)
rhel4           Red Hat Enterprise Linux 4 with updates on x86_64 (64 bit mode)
rh5             Red Hat Enterprise Linux 5 with updates on ia32
rhel5           Red Hat Enterprise Linux 5 with updates on x86_64 (64 bit mode)
rhel51          Red Hat Enterprise Linux 5.1 with updates on x86_64 (64 bit mode)
rhel52          Red Hat Enterprise Linux 5.2 with updates on x86_64 (64 bit mode)
rhel53          Red Hat Enterprise Linux 5.3 with updates on x86_64 (64 bit mode)
rhel6           Red Hat Enterprise Linux 6 with updates on x86_64 (64 bit mode)
rhel7           Red Hat Enterprise Linux/CentOS 7 with updates on x86_64 (64 bit mode)

teck            Fedora Linux Core 3 with updates on ia32
gcc4            Fedora Linux Core 4 with updates on ia32
heiner          Fedora Linux Core 5 with updates on ia32
belchen         Fedora Linux Core 6 with updates on ia32
monsruemker     Fedora Linux Core 7 with updates on ia32
neuffen         Fedora Linux Core 8 with updates on ia32
stromboli       Fedora Linux Core 9 with updates on ia32

amd64           Fedora Linux Core 3 with updates on x86_64
amd64icc        Fedora Linux Core 3 with updates on x86_64 and Intel Compiler 8.1
x64             Fedora Linux Core 4 with updates on x86_64
bishorn         Fedora Linux Core 5 with updates on x86_64
fujisan         Fedora Linux Core 6 with updates on x86_64
monshuygens     Fedora Linux Core 7 with updates on x86_64
lycaeus         Fedora Linux Core 8 with updates on x86_64
maunaloa        Fedora Linux Core 9 with updates on x86_64
gorely          Fedora Linux Core 10 with updates on x86_64
leonidas        Fedora Linux Core 11 with updates on x86_64
constantine     Fedora Linux Core 12 with updates on x86_64
goddard         Fedora Linux 13 with updates on x86_64
laughlin        Fedora Linux 14 with updates on x86_64
lovelock        Fedora Linux 15 with updates on x86_64
verne           Fedora Linux 16 with updates on x86_64

chuckwalla      SUSE 9.2 on ia32
lurchi          SUSE 9.3 on ia32
leguan          SUSE 9.3 on x86_64

gecko           OpenSUSE 10.0 on ia32
skink           OpenSUSE 10.1 on ia32
dornteufel      OpenSUSE 10.2 on ia32
agame           OpenSUSE 10.3 on ia32
tiliqua         OpenSUSE 11.0 on ia32
waran           OpenSUSE 10.0 on x86_64
basilisk        OpenSUSE 10.1 on x86_64 / SUSE Enterprise Server 10 on x86_64
iguana          OpenSUSE 10.2 on x86_64
tuatara         OpenSUSE 10.3 on x86_64
mabuya          OpenSUSE 11.0 on x86_64
drusenkopf      OpenSUSE 11.1 on x86_64
lipinia         OpenSUSE 11.2 on x86_64
mamba           OpenSUSE 11.3 on x86_64
indicus         OpenSUSE 12.1 on x86_64
slowworm        OpenSUSE 12.2 on x86_64
neolamprologus  OpenSUSE 12.3 on x86_64
saara           OpenSUSE 13.1 on x86_64
julidochromis   OpenSUSE 13.2 on x86_64
cyprichromis    OpenSUSE Leap 42.2 on x86_64
tangachromis    openSUSE Leap 15.1
altolamprologus openSUSE Leap 15.2
leap153         openSUSE Leap 15.3
leap154         openSUSE Leap 15.4
chalinochromis  openSUSE Tumbleweed

sarge           Debian Linux 3.1 (Sarge) with updates on ia32
etch32          Debian Linux 4.0 (Etch) with updates on ia32
etch            Debian Linux 4.0 (Etch) with updates on x86_64
lenny32         Debian Linux 5.0 (Lenny) with updates on ia32
lenny           Debian Linux 5.0 (Lenny) with updates on x86_64
squeeze         Debian Linux 6.0 (Squeeze) with updates on x86_64
buster          Debian Linux 10 (Buster) with updates on x86_64

dapper          Ubuntu 6.06 Dapper Drake on ia32
edgy            Ubuntu 6.10 Edgy Eft on ia32
feisty          Ubuntu 7.04 Feisty Fawn on ia32
gutsy           Ubuntu 7.10 Gutsy Gibbon on ia32
hardy           Ubuntu 8.04 Hardy Heron on ia32
intrepid        Ubuntu 8.10 Intrepid Ibex on ia32
jaunty          Ubuntu 9.04 Intrepid Ibex on ia32
karmic          Ubuntu 9.10 Intrepid Ibex on ia32
lucid           Ubuntu 10.04 Lucid Lynx on ia32
maverick        Ubuntu 10.10 Maverick Meerkat on ia32
natty           Ubuntu 11.04 Natty Narwhal on ia32
oneiric         Ubuntu 11.10 Natty Narwhal on ia32
precise		    Ubuntu 12.04 LTS Precise Pangolin on ia32
drake           Ubuntu 6.06 Dapper Drake on x86_64 (64 bit mode)
eft             Ubuntu 6.10 Edgy Eft on x86_64 (64 bit mode)
fawn            Ubuntu 7.04 Feisty Fawn on x86_64 (64 bit mode)
gibbon          Ubuntu 7.10 Gutsy Gibbon on x86_64 (64 bit mode)
heron           Ubuntu 8.04 Hardy Heron on x86_64 (64 bit mode)
ibex            Ubuntu 8.10 Intrepid Ibex on x86_64 (64 bit mode)
jackalope       Ubuntu 9.04 Intrepid Ibex on x86_64 (64 bit mode)
koala           Ubuntu 9.10 Intrepid Ibex on x86_64 (64 bit mode)
lynx            Ubuntu 10.04 Lucid Lynx on x86_64 (64 bit mode)
meerkat         Ubuntu 10.10 Maverick Meerkat on x86_64 (64 bit mode)
narwhal         Ubuntu 11.04 Natty Narwhal on x86_64 (64 bit mode)
ocelot          Ubuntu 11.10 Oneiric Ocelot on x86_64 (64 bit mode)
pangolin        Ubuntu 12.04 Precise Pangolin on x86_64 (64 bit mode)
tahr            Ubuntu 14.04 Trusty Tahr and Linux Mint 17 on x86_64 (64 bit mode)
vervet          Ubuntu 15.04 Vivid Vervet on x86_64 (64 bit mode)
werewolf        Ubuntu 15.10 Wily Werewolf on x86_64 (64 bit mode)
xerus           Ubuntu 16.04 Xenial Xerus on x86_64 (64 bit mode)
bionic          Ubuntu 18.04 Bionic Beaver on x86_64 (64 bit mode)
focal           Ubuntu 20.04 Focal Fossa on x86_64 (64 bit mode)

macx            Mac OS X 10.3, PPC
tiger           Mac OS X 10.4
osx11           Mac OS X 10.4 on ia32 with X11 and fink
leopard         Mac OS X 10.5 or 10.6, Universal
lion            Mac OS X 10.7-10.9, x86_64 with libstdc++
libc++          Mac OS X 10.9-10.12, x86_64 with libc++
