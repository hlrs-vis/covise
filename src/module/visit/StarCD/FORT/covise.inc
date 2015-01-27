C --- Covise common block definition file

C --- we need to know some parameters of the compilation
      include '../parm.inc'

C     WARNING: these dimensions have to be complient to the module
      PARAMETER (MAXREG = 16)
      PARAMETER (MAXSCA = 10)
      PARAMETER (MAXUSR = 10)
      PARAMETER (MAXOUT = 6)
C     if MAXOUT is changed, also change DATA line with initialisation
      
C --- This is the common block --------------------------------------------------------
C
C    int   iconum;              // number of regions
C
C    Flags part: 1 flag per region
C
C    int32 icoreg[MAX_REGIONS]; // region number:  0 if not set
C    int32 icovel[MAX_REGIONS]; // = 0 if velocity     not set, =1  if set
C    int32 icot  [MAX_REGIONS]; // = 0 if temperature  not set, =1  if set
C    int32 icoden[MAX_REGIONS]; // = 0 if density      not set, =1  if set
C    int32 icotur[MAX_REGIONS]; // = 0 if turbulence   not set, =1  k/eps, =-1 Int/len
C    int32 icop  [MAX_REGIONS]; // = 0 if pressure     not set, =1  if set
C    int32 icosca[MAX_REGIONS][MAX_SCALARS]; // 0 for none  # for used scalar
C
C    Value part: Values only read once per iteration for all used parameters
C
C     -> corresponding to flag fields, tu1/tu2  either k/eps or Int/len
C        
C    IFIRST   used only once to determine whether we ever tried to connect
C    ICOBOC   last iteration we requested our parameters
C    IREGNO   backward mapping of region numbers: 
C             IREGNO(7)=4 means we set parameter for region 7 in our xxx[4] in common blk
C    icostp   transmitted with boco: steps to go
C             
C
CC --------------------------------------------------------------------------------------        


      COMMON/COVISE/iconum,icoreg(MAXREG),icovel(MAXREG),icot(MAXREG),
     +              icoden(MAXREG),icotur(MAXREG),icop(MAXREG),
     +              icosca(MAXSCA,MAXREG),icousr(MAXUSR,MAXREG),
     +
     +              icostp,icoout(MAXOUT),
     +              covvel(3,MAXREG),covt(MAXREG),covden(MAXREG),
     +              covtu1(MAXREG),covtu2(MAXREG),covp(MAXREG),
     +              covsca(MAXSCA,MAXREG),covusr(MAXUSR,MAXREG),
     +
     +              IFIRST,ICOBOC,iregno(NREGMX),coport(MAXOUT),
     +              icomat

      INTEGER*4 iconum,icoreg,icovel,icot,icoden,icotur,icop,
     +          icosca,icostp,icoout
     
      REAL*4    covvel,covt,covden,covtu1,covtu2,covp,covsca
      CHARACTER*8 coport 

C --- Number of BYTES to be transferred at start-up
      PARAMETER ( ISETUP = 4*(1+MAXREG*(6+MAXSCA+MAXUSR))        )
      PARAMETER ( IPARAM = 4*(1+MAXREG*(8+MAXSCA+MAXUSR)+MAXOUT) ) 
      
C --- length of init part in common block (ints only)
      PARAMETER ( INITCO = 1+MAXREG*(6+MAXSCA+MAXUSR) )

C --- length of BC common block (ints, floats)
      PARAMETER ( IBCLIN = 1+MAXOUT )
      PARAMETER ( IBCLFL = MAXREG*(8+MAXSCA+MAXUSR) )

C --- length of BC part in common (ints only)
      
C --- this is stuff for the parallel version only
#ifdef PARALLEL_STAR
      INCLUDE   'mpif.h'
      INTEGER*4 LOC_TO_GLOBAL(NCTMAX,MAXDMN),ICOUNT(MAXDMN)
c     MPI status field: STAT(MPI_TAG), STAT(MPI_SOURCE)
c     retrieve length with MPI_Get_count(stat,<type>,COUNT,info)
      INTEGER ISTAT(MPI_STATUS_SIZE)
      COMMON /MPP1/ NOPROC,MYID
      common /compos/LOC_TO_GLOBAL,ICOUNT,NCELL_G,MAXCOUNT,UDATA

c     minimum tag number to use      
      PARAMETER ( MTAGU = 30000 )

#endif
c      REAL*4 UDATA_COPY(NCTMAX)
c      REAL*4 UDATA(NCTMAX*MAXDMN)   
