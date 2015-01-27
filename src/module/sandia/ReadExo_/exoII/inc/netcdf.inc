c      
c     Functions in the FORTRAN interface

      integer nccre, ncopn, ncddef, ncdid, ncvdef, ncvid, nctlen
      integer ncsfil
      external nccre, ncopn, ncddef, ncdid, ncvdef, ncvid, nctlen
      external ncsfil

      integer NCRDWR,NCCREAT,NCEXCL,NCINDEF,NCNSYNC,NCHSYNC
      integer NCNDIRTY,NCHDIRTY,NCLINK,NCNOWRIT,NCWRITE
      integer NCCLOB,NCNOCLOB,NCGLOBAL,NCFILL,NCNOFILL
      integer MAXNCOP,MAXNCDIM,MAXNCATT,MAXNCVAR
      integer MAXNCNAM,MAXVDIMS,NCNOERR,NCEBADID
      integer NCENFILE,NCEEXIST,NCEINVAL,NCEPERM,NCENOTIN
      integer NCEINDEF,NCECOORD,NCEMAXDS,NCENAME
      integer NCENOATT,NCEMAXAT,NCEBADTY,NCEBADD, NCESTS
      integer NCEUNLIM,NCEMAXVS,NCENOTVR,NCEGLOB,NCENOTNC
      integer NCFOOBAR,NCSYSERR,NCFATAL,NCVERBOS, NCENTOOL
      
      integer NCBYTE,NCCHAR,NCSHORT,NCLONG,NCFLOAT,NCDOUBLE
      
      parameter(NCBYTE = 1)
      parameter(NCCHAR = 2)
      parameter(NCSHORT = 3)
      parameter(NCLONG = 4)
      parameter(NCFLOAT = 5)
      parameter(NCDOUBLE = 6)
      
c     
c     masks for the struct NC flag field; passed in as 'mode' arg to
c     nccreate and ncopen.
c     
      
c     read/write, 0 => readonly 
      parameter(NCRDWR = 1)
c     in create phase, cleared by ncendef 
      parameter(NCCREAT = 2)
c     on create destroy existing file 
      parameter(NCEXCL = 4)
c     in define mode, cleared by ncendef 
      parameter(NCINDEF = 8)
c     synchronise numrecs on change (X'10')
      parameter(NCNSYNC = 16)
c     synchronise whole header on change (X'20')
      parameter(NCHSYNC = 32)
c     numrecs has changed (X'40')
      parameter(NCNDIRTY = 64)	
c     header info has changed (X'80')
      parameter(NCHDIRTY = 128)
c     prefill vars on endef and increase of record, the default behavior
      parameter(NCFILL = 0)
c     don't fill vars on endef and increase of record (X'100')
      parameter(NCNOFILL = 256)
c     isa link (X'8000')
      parameter(NCLINK = 32768)
      
c     
c     'mode' arguments for nccreate and ncopen
c     
      
      parameter(NCNOWRIT = 0)
      parameter(NCWRITE = NCRDWR)
      parameter(NCCLOB = 11)
      parameter(NCNOCLOB = 15)
c     
c     'size' argument to ncdimdef for an unlimited dimension
c     
      integer NCUNLIM
      parameter(NCUNLIM = 0)
      
c     
c     attribute id to put/get a global attribute
c     
      parameter(NCGLOBAL  = 0)
c     
c     Advisory Maximums
c     
      parameter(MAXNCOP = 32)
      parameter(MAXNCDIM = 32)
      parameter(MAXNCATT = 512)
      parameter(MAXNCVAR = 512)
c     Not enforced 
      parameter(MAXNCNAM = 128)
      parameter(MAXVDIMS = MAXNCDIM)
      
      
c     
c     The netcdf data types
c     
      
c     
c     Global netcdf error status variable
c     Initialized in error.c
c     
      
c     No Error 
      parameter(NCNOERR = 0)
c     Not a netcdf id 
      parameter(NCEBADID = 1)
c     Too many netcdfs open 
      parameter(NCENFILE = 2)	
c     netcdf file exists && NCNOCLOB
      parameter(NCEEXIST = 3)
c     Invalid Argument 
      parameter(NCEINVAL = 4)
c     Write to read only 
      parameter(NCEPERM = 5)
c     Operation not allowed in data mode 
      parameter(NCENOTIN = 6)	
c     Operation not allowed in define mode 
      parameter(NCEINDEF = 7)	
c     Coordinates out of Domain 
      parameter(NCECOORD = 8)
c     MAXNCDIMS exceeded 
      parameter(NCEMAXDS = 9)
c     String match to name in use 
      parameter(NCENAME = 10)	
c     Attribute not found 
      parameter(NCENOATT = 11)
c     MAXNCATTRS exceeded 
      parameter(NCEMAXAT = 12)
c     Not a netcdf data type 
      parameter(NCEBADTY = 13)
c     Invalid dimension id 
      parameter(NCEBADD = 14)	
c     NCUNLIMITED in the wrong index 
      parameter(NCEUNLIM = 15)
c     MAXNCVARS exceeded 
      parameter(NCEMAXVS = 16)
c     Variable not found 
      parameter(NCENOTVR = 17)
c     Action prohibited on NCGLOBAL varid 
      parameter(NCEGLOB = 18)
c     Not a netcdf file 
      parameter(NCENOTNC = 19)
      parameter(NCESTS = 20)
      parameter (NCENTOOL = 21)	
      parameter(NCFOOBAR = 32)
      parameter(NCSYSERR = -1)
      
      
c     
c     Global options variable. Used to determine behavior of error handler.
c     Initialized in lerror.c
c     
      parameter(NCFATAL = 1)
      parameter(NCVERBOS = 2)
