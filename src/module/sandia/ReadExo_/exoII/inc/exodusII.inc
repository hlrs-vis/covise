C****************************************************************
C
C Name:	exodusII.inc - Exodus II Application include file 
C
C	Use this include file to define general Exodus II parameters
C
C author - Victor R. Yarberry, Sandia National Laboratories
C
C environment - UNIX F77
C
C revision history - 
C
C History:
C  02/25/93	V.R. Yarberry, Original derived from exodus.inc
C
C Revision 1.3  1994/03/30  16:06:35  mksmith
C Upgrade exodusII version 2.00 to version 2.01
C
C Revision 1.2  1993/10/25  17:05:23  vryarbe
C updated with new defines
C
C Revision 1.1  1993/09/08  17:06:16  vryarbe
C added new include file
C
C
C*****************************************************************************
C
C	Define Exodus functions that return values
C
	integer excre,exopen
C
C       Define access modes
C
	integer EXNOCL,EXCLOB,EXREAD,EXWRT
C	EXNOCL - File create with no overwrite (no clobber)
	parameter (EXNOCL = 0)
C	EXCLOB - File create with overwrite (clobber)
	parameter (EXCLOB = 1)
C	EXREAD - File open for read
	parameter (EXREAD = 0)
C	EXWRT  - File open for write
	parameter (EXWRT = 1)
C
C  --INQUIRE FLAGS
C  --
C  --   EXTYPE - EXODUS II file type is desired.
C  --   EXVERS - EXODUS API version number (in file) is desired
C  --   EXDBVR - EXODUS Database version number is desired
C  --   EXLBVR - EXODUS API Library version number is desired
C  --   EXTITL - Netcdf file title is desired
C  --   EXDIM  - Dimensionality (the number of coordinates per node)
C  --            is desired.
C  --   EXNODE - The number of nodal points is desired.
C  --   EXELEM - The number of elements is desired.
C  --   EXELBL - The number of element blocks is desired.
C  --   EXNODS - The number of node sets is desired.
C  --   EXSSNL - The length of the concatenated node list for all the
C  --            node sets.
C  --   EXSIDS - The number of side sets is desired.
C  --   EXSSEL - The length of the concatenated element list for all
C  --            the side sets.
C  --   EXNSNL - The length of the concatenated node list for all the
C  --            side sets.
C  --   EXQA   - The number of QA records is desired.
C  --   EXINFO - The number of information records is desired.
C  --   EXTIMS - The number of time steps is desired.
C  --   EXNEBP - The number of element block properties
C  --   EXNNSP - The number of node set properties
C  --   EXNSSP - The number of side set properties
C  --   EXNSDF - The length of the node set dist factors array
C  --   EXSSDF - The length of the side set dist factors array
C  --
      integer EXTYPE, EXVERS, EXDBVR, EXTITL, EXDIM, EXNODE,
     1        EXELEM, EXELBL, EXNODS, EXSIDS, EXQA, EXINFO,
     2        EXTIMS, EXSSNL, EXSSEL, EXNSNL, EXNEBP, EXNNSP,
     3        EXNSSP, EXNSDF, EXSSDF, EXLBVR

c
      parameter (EXTYPE = 1)
      parameter (EXVERS = 2)
      parameter (EXDBVR = 3)
      parameter (EXTITL = 4)
      parameter (EXDIM  = 5)
      parameter (EXNODE = 6)
      parameter (EXELEM = 7)
      parameter (EXELBL = 8)
      parameter (EXNODS = 9)
      parameter (EXNSNL = 10)
      parameter (EXSIDS = 11)
      parameter (EXSSNL = 12)
      parameter (EXSSEL = 13)
      parameter (EXQA   = 14)
      parameter (EXINFO = 15)
      parameter (EXTIMS = 16)
      parameter (EXNEBP = 17)
      parameter (EXNNSP = 18)
      parameter (EXNSSP = 19)
      parameter (EXNSDF = 20)
      parameter (EXSSDF = 21)
      parameter (EXLBVR = 22)
C
C         Define limits
C
	integer MXSTLN, MXLNLN
C	MXSTLN - Maximum string length: used for store strings
	parameter (MXSTLN = 32)
C	MXLNLN - Maximum line string length
	parameter (MXLNLN = 80)
C
C	properties
C
	integer EXEBLK, EXNSET, EXSSET
C	EXEBLK - element block property code
	parameter (EXEBLK = 1)
C	EXNSPR - node set property code
	parameter (EXNSET = 2)
C	EXSSPR - side set property code
	parameter (EXSSET = 3)
C
C	Define options codes
C
	integer EXVRBS, EXDEBG, EXABRT
C	EXVRBS - verbose mode message flag
	parameter (EXVRBS = 1)
C	EXDEBG - debug mode message flag
	parameter (EXDEBG = 2)
C	EXABRT - abort mode flag
	parameter (EXABRT = 4)
C
C	Define error return codes
C
	integer EXFATL, EXOK, EXWARN
C	EXFATL - fatal error flag def
	parameter (EXFATL = -1)
C	EXOK - no err (success) flag def
	parameter (EXOK = 0)
C	EXWARN - warning flag def
	parameter (EXWARN = 1)
C
C	Define error return codes and flags
C
        integer EXMEMF, EXBFMD, EXBFID, EXBTID, EXBPRM, EXPMSG,
     1          EXLMSG
C       EXMEMF - memory allocation failure flag def
        parameter (EXMEMF = -100)
C       EXBFMD - wrong file mode
        parameter (EXBFMD = -101)
C       EXBFID - bad file id def
        parameter (EXBFID = -102)
C       EXBTID - id table lookup failed
        parameter (EXBTID = -104)
C       EXBPRM - bad parameter passed
        parameter (EXBPRM  =  -105)
C       EXPMSG - message print code - no error implied
        parameter (EXPMSG = 100)
C       EXLMSG - print last error message msg code
        parameter (EXLMSG = 101)
