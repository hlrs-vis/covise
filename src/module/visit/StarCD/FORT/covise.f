#define VERBOSE
#define PARALLEL_STAR

C*************************************************************************
C*************************************************************************
C*************************************************************************
C*************************************************************************
C********                                                         ********
C********      #####           #####                              ********
C********     #     #   ####  #     #   #####    ##    #####      ********
C********     #        #    # #           #     #  #   #    #     ********
C********     #        #    #  #####      #    #    #  #    #     ********
C********     #        #    #       #     #    ######  #####      ********
C********     #     #  #    # #     #     #    #    #  #   #      ********
C********      #####    ####   #####      #    #    #  #    #     ********
C********                                                         ********
C*******                                                           *******      
C******* SUB for startup proc: either called from BCDEFI or BCDEFP *******
C*******                                                           *******      
C*************************************************************************
C*************************************************************************
C*************************************************************************
C*************************************************************************


      
      SUBROUTINE COSTAR(ICLMAP,KEY)
            
C --- declares Simlib, Covise common-block and sizes
      INCLUDE 'comdb.inc'
      INCLUDE 'coSimLib.inc'
      INCLUDE 'covise.inc'
      INCLUDE 'usrdat.inc'
      DIMENSION ICLMAP(NCTMXU),KEY(-NBMAXU:NCTMXU)

C --- pre-set some values
      DATA IFIRST /1/
      DATA ICOBOC /-1/
      DATA ICOSTP /1/
      DATA coport/'data_0','data_1','data_2','data_3','data_4','data_5'/
      
C --- we need this to transfer the mappings C=Covise, S=Star
      INTEGER*4 IMBUFC(NCTMAX)
      INTEGER   IMBUFS(NCTMAX)

C --- find maximum material number      
      
      icomat=0
      DO 10 i=1,NCTMXU
  10     if (KEY(i).gt.icomat) icomat=key(i)

c      write(*,*) 'Local max:',icomat

#ifdef PARALLEL_STAR 

C     in the parallel case, we merge all-to-all
      call MPI_Allreduce(icomat,ihelp,1,MPI_INTEGER,MPI_MAX,
     +                   MPI_COMM_WORLD,ierr)
      icomat=ihelp

      if (MYID.eq.1) then

#endif

C ####################################################################
C     Master / Sequential startup
C #################################################################### 

         if (COVINI().NE.0) then
            write(*,*) ''
            write(*,*) '  *******************************************'
            write(*,*) '****        COULD NOT CONNECT : QUIT       ****'
            write(*,*) '  *******************************************'
            write(*,*) ''
            i=-1
#ifdef PARALLEL_STAR 
            call MPI_BCAST(i,1,MPI_INTEGER,0,MPI_COMM_WORLD,INFO)
            call MPI_FINALIZE(info)
#endif
            stop
            
         else
            write(*,*) ''
            write(*,*) '  *******************************************'
            write(*,*) '****          CONNECTED TO COVISE          ****'
            write(*,*) '  *******************************************'
            write(*,*) ''
         endif
         
C ---    Receive flags block

         if (CORECV(iconum,ISETUP).ne.ISETUP) then
            write(*,*) ''
            write(*,*) '  *******************************************'
            write(*,*) '****  Common set-up not received : QUIT    ****'
            write(*,*) '  *******************************************'
            write(*,*) ''
            i=-1
#ifdef PARALLEL_STAR 
            call MPI_BCAST(i,1,MPI_INTEGER,0,MPI_COMM_WORLD,INFO)
            call MPI_FINALIZE(info)
#endif
            stop
         endif

C ---    parallel or not, we send some info to covise

#ifdef PARALLEL_STAR
c        parallel: number of nodes - number of cells comes from mappings
         IMBUFC(1) = NOPROC
#else
c        sequential : number of cells - negative to be able to discriminate
         IMBUFC(1) = -NCELL
#endif
         if (COSEND(IMBUFC,4).ne.4) then
            write(*,*) ''
            write(*,*) '  *******************************************'
            write(*,*) '****   Could not send to covise : QUIT     ****'
            write(*,*) '  *******************************************'
            write(*,*) ''
            i=-1
#ifdef PARALLEL_STAR 
            call MPI_BCAST(i,1,MPI_INTEGER,0,MPI_COMM_WORLD,INFO)
            call MPI_FINALIZE(info)
#endif
            stop
         endif

#ifdef PARALLEL_STAR 

C ---    Send my own local-to-global block to covise
C        do not use automatic mapping routines, these are PROSTAR numbers
C ---    copy to INTEGER*4 buffer to assure same format as receiver
         
         
C ---    make length int32 and send it
         IMBUFC(1) = NCELL
         if (COSEND(IMBUFC,4).ne.4) THEN
            write(*,*) '****  Failed send Mapping length node 1 ****'
            i=-1
            call MPI_BCAST(i,1,MPI_INTEGER,0,MPI_COMM_WORLD,INFO)
            call MPI_FINALIZE(info)
            stop
         endif

C ---    make length int32 and send it
         do 20 i=1,NCELL
            IMBUFC(i) = ICLMAP(i)
   20    continue
         if (COSEND(IMBUFC,NCELL*4).ne.NCELL*4) THEN
            write(*,*) '****  Failed sending Mapping of node 1  ****'
            i=-1
            call MPI_BCAST(i,1,MPI_INTEGER,0,MPI_COMM_WORLD,INFO)
            call MPI_FINALIZE(info)
         endif
         
C ---    tell the slaves we are ok
         i=1
         call MPI_BCAST(i,1,MPI_INTEGER,0,MPI_COMM_WORLD,INFO)

C ---    sync before scatter data transfer
         call MPI_Barrier(MPI_COMM_WORLD,info)
         
C ---    relay slave's info to Covise   
         do 40 i=1,NOPROC-1

             call MPI_Recv(IMBUFS,NCTMAX,MPI_INTEGER,i,MPI_ANY_TAG,
     +                     MPI_COMM_WORLD,istat,info)
C ---       get length, make it int32 and send it
            call MPI_Get_count(istat,MPI_INTEGER,INUM,info)

            IMBUFC(1) = INUM
            if (COSEND(IMBUFC,4).ne.4) THEN
               write(*,*) '****  Failed sending Mapping length of node'
     +                    ,j,'  ****'
            endif

C ---       make field int32 and send it
            do 30 j=1,INUM
               IMBUFC(j) = IMBUFS(j)
   30       continue
            if (COSEND(IMBUFC,INUM*4).ne.INUM*4) THEN
               write(*,*) '****  Failed sending Mapping of node'
     +                    ,j,'  ****'
            endif
   40    continue

C ####################################################################
C     Slave startup in parallel case
C #################################################################### 
       
      else
 
C ---    Got connection? - otherwise finalize and exit 
         call MPI_BCAST(i,1,MPI_INTEGER,0,MPI_COMM_WORLD,INFO)
         if (i.lt.0) then
            call MPI_FINALIZE(info)
            stop
         endif            

C ---    sync before scatter data transfer
         call MPI_Barrier(MPI_COMM_WORLD,info)


         call MPI_SEND(ICLMAP,NCELL,MPI_INTEGER,0,MTAGU,
     +                 MPI_COMM_WORLD,info)
      endif

C ####################################################################
C     Master & slave common startup parts
C #################################################################### 
       

C --- sync after scatter data transfer
      call MPI_Barrier(MPI_COMM_WORLD,info)

C --- ALL together again: share the FLAGS block
      call MPI_BCAST(iconum,INITCO,MPI_INTEGER,0,MPI_COMM_WORLD,INFO)
#endif

C --- Now everybody knows: we are connected 
      IFIRST=0


C --- Create backward mapping for Region numbers - faster for per-cell ops      
      
      do 50 i=1,NREGMX
         iregno(i) = 0
 50   continue
      
      do 60 i=1,iconum
         iregno(icoreg(i)) = i
 60   continue
 
C --- later calls to COBREC are done from POSDAT level=2
C     so we need to reduce the number of steps here
      icostp = icostp-1



C --- now receive our first BC block so we can create data...
      call COBREC()
      
      RETURN
      END


C*************************************************************************
C*************************************************************************
C*************************************************************************
C*************************************************************************
C********                                                         ********
C********     .#####.         ######.                             ********
C********     #     #  .####. #     #   ####.  .####,   #####     ********
C********     #        #    # #     #  #    #  #          #       ********
C********     #        #    # ######   #    #  `####.     #       ********
C********     #        #    # #        #    #       #     #       ********
C********     #     #  #    # #        #    #  #    #     #       ********
C********     `#####'  `####' #         ####   `####      #       ********
C********                                                         ********
C*************************************************************************
C*************************************************************************
C*************************************************************************
C*************************************************************************

      SUBROUTINE COPOST(KEY,VOL,U,TE,ED,T,P,VIST,DEN,CP,VISM,CON,
     *  F,ICLMAP,ICTID,RESOR,VF,FORCB,IRN,PREFM,LEVEL,
     *  NCOUS1,COVUS1,NCOUS2,COVUS2)

      INCLUDE 'comdb.inc'
      
C --- declares Simlib, Covise common-block and sizes
      INCLUDE 'coSimLib.inc'
      INCLUDE 'covise.inc'

      COMMON/USR001/INTFLG(100)

      DIMENSION KEY(-NBMAXU:NCTMXU),VOL(NCTMXU),U(3,-NBMAXU:NCMAXU),
     * TE(-NBMAXU:NCMAXU),ED(-NBMAXU:NCMAXU),T(-NBMAXU:NCTMXU,1+NSCU),
     * P(-NBMAXU:NCMAXU),VIST(-NBMAXU:NCMAXU),DEN(-NBMAXU:NCTMXU),
     * CP(-NBMAXU:NCTMXU),VISM(-NBMXVU:NCMXVU),CON(-NBMXCU:NCMXCU),
     * F(3,-NBMAXU:NCMAXU),ICLMAP(NCTMXU),ICTID(NCTMXU),
     * RESOR(63,-100:100),VF(NCDMXU),
     * FORCB(3,NWLMX),IRN(NWLMX)
      DOUBLE PRECISION P
      DIMENSION PREFM(100)
      CHARACTER*80 REALTI
      INCLUDE 'usrdat.inc'

C ========================================================================
C --- Two user-supplied fields: if not needed, put in any field as dummy
      REAL*4  COVUS1(NCTMXU),COVUS2(NCTMXU)
      INTEGER NCOUS1,NCOUS2

C --- Our 'collect' array
C      must be as big as largest node's dimension      @@@ MPP @@@
      REAL*4 udata(NCTMAX,3)
      
C ======================================================================
C ==== We have to create output now
C ======================================================================

C --- it may not be out turn yet
      if (ITER.ge.ICOBOC) then 

C ---    we have to send an EXEC msg to the module

#ifdef PARALLEL_STAR 
         if (MYID.eq.1) then
#endif
            if (COEXEC().ne.0) then
               write(*,*) '  *************************************'
               write(*,*) '**** COEXEC call not succeeded: exit ****'
               write(*,*) '  *************************************'
               open(unit=99,file="ABORT")
               write(99,*) "Abort"
               close(99)
            endif

c         write(*,*) PREFM(1),PREFM(2),PREFM(3),PREFM(4),PREFM(5)

#ifdef PARALLEL_STAR 
         endif

C ---    In the parallel version, we have to copy the residuals and
C        perform some collective operation on it 
         
#else

#endif


         do 1000 i=1,MAXOUT,1

            ICOMP=1
            if (icoout(i).lt.20) then
               goto (1000,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18)
     +           ,icoout(i)
c           -------------------------------------------------------------           
  2            continue
                 ICOMP=3
                 INUM=NCELL
C                special case: vector     
                  DO 21 j=1,NCELL
                     udata(j,1)=U(1,j)
                     udata(j,2)=U(2,j)
  21                 udata(j,3)=U(3,j)
                  continue
               goto 999
c           -------------------------------------------------------------           
  3            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 31 j=1,NCELL
                     udata(j,1)=SQRT( U(1,j)*U(1,j)
     +                               +U(2,j)*U(2,j)
     +                               +U(3,j)*U(3,j) )
 31               continue
               goto 999
c           -------------------------------------------------------------           
  4            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 41 j=1,NCELL
                     udata(j,1)=U(1,j)
 41               continue
               goto 999
c           -------------------------------------------------------------           
  5            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 51 j=1,NCELL
                     udata(j,1)=U(2,j)
 51               continue
               goto 999
c              -------------------------------------------------------------
  6            continue
                  DO 61 j=1,NCELL
                  ICOMP=1
                  INUM=NCELL
                     udata(j,1)=U(3,j)
 61               continue
               goto 999
c              -------------------------------------------------------------
  7            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 71 j=1,NCELL
                     udata(j,1)=P(j)
 71               continue
               goto 999
c              -------------------------------------------------------------
  8            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 81 j=1,NCELL
                     udata(j,1)=TE(j)
 81               continue
               goto 999
c              -------------------------------------------------------------
  9            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 91 j=1,NCELL
                     udata(j,1)=ED(j)
 91               continue
               goto 999
c              -------------------------------------------------------------
 10            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 101 j=1,NCELL
                     udata(j,1)=VIST(j)
101               continue
               goto 999

c              -------------------------------------------------------------
 11            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 111 j=1,NCELL
                     udata(j,1)=T(j,1)
111               continue
               goto 999
c              -------------------------------------------------------------
 12            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 121 j=1,NCELL
                     udata(j,1)=DEN(j)
121               continue
               goto 999
c              -------------------------------------------------------------
 13            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 131 j=1,NCELL
                     udata(j,1)=VISM(j)
131               continue
               goto 999
c              -------------------------------------------------------------
 14            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 141 j=1,NCELL
                     udata(j,1)=CP(j)
141               continue
               goto 999
c              -------------------------------------------------------------
 15            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 151 j=1,NCELL
                     udata(j,1)=CON(j)
151               continue
               goto 999
C              -------------------------------------------------------------
C              FLUX
 16            continue
                 ICOMP=3
                 INUM=NCELL
C                special case: vector     
                  DO 22 j=1,NCELL
                     udata(j,1)=F(1,j)
                     udata(j,2)=F(2,j)
  22                 udata(j,3)=F(3,j)
                  continue
               goto 999
C              -------------------------------------------------------------
C              VOID Fraction
 17            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 171 j=1,NCELL
                     udata(j,1)=VF(j)
171               continue
               goto 999
c              -------------------------------------------------------------
 18            continue
                  ICOMP=1
                  INUM=NCELL
                  DO 181 j=1,NCELL
                     udata(j,1)=VOL(j)
181               continue
               goto 999
c              -------------------------------------------------------------

c               #20 and more are SCALARs : Scalar1 = #20 = T(i,2)

            else

               ICOMP=1
               INUM=NCELL
               DO 501 j=1,NCELL
                  udata(j,1)=T(j,icoout(i)-18)
501            continue

            endif

C ---    now we crate the data object

 999     continue

#ifdef PARALLEL_STAR 
         if (MYID.eq.1) then

c           tell Covise that node 0 (FORTRAN 1) is sending now
            call COPANO(0)
#endif


            if (ICOMP.eq.3) then
               if (COSU3D(coport(i),INUM,
     +             udata(1,1),udata(1,2),udata(1,3)).ne.0) then
                  write(*,*) 'PROBLEMS creating DO for',coport(i)
               endif
            else
               if (COSU1D(coport(i),INUM,udata(1,1)).ne.0) then
                  write(*,*) 'PROBLEMS creating for',coport(i)
               endif
            endif
            
c            write(*,*) 'success',coport(i)

c ---    this is where we carry on after sending the data
 998        continue
#ifdef PARALLEL_STAR 

c ---    on MPP, we now hand through all slave node's field to COVISE

C           Synchronize now - remove when knowing about 'user' TAGs
            call MPI_Barrier(MPI_COMM_WORLD,info)
         
            do 400 ip=1,NOPROC-1
         
c              tell Covise that node i (FORTRAN i-1) is sending now
               call COPANO(ip)
               
c              -- should be working async: get the first, look who ...
               do 410 k=1,ICOMP 
c      write(*,*) 'MPI_RECV Length=',NCTMAX
                  call MPI_RECV(udata(1,k),NCTMAX,MPI_REAL,ip,k+MTAGU,
     +                     MPI_COMM_WORLD,istat,info)
 410           continue
 
C              get size of data field
               call MPI_Get_count(istat,MPI_REAL,INUM,info)

               if (icomp.eq.3) then
                  if (COSU3D(coport(i),INUM,
     +                udata(1,1),udata(1,2),udata(1,3)).ne.0) then
                     write(*,*) 'PROBLEMS creating DO for',coport(i)
                  endif
               else
                  if (COSU1D(coport(i),INUM,udata(1,1)).ne.0) then
                     write(*,*) 'PROBLEMS creating DO for',coport(i)
                  endif
               endif


 400        continue

C           Synchronize now - remove when knowing about 'user' TAGs
            call MPI_BARRIER(MPI_COMM_WORLD,info)

         else

C ---    This is a slave node : send my field to the master 
C                               use tags to ensure correct order

C           Synchronize now
            call MPI_Barrier(MPI_COMM_WORLD,info)
         
            
            do 500 k=1,ICOMP 
               call MPI_SEND(udata(1,k),NCELL,MPI_REAL,0,k+MTAGU,
     +                       MPI_COMM_WORLD,info)
 500        continue
         
C           Synchronize now
            call MPI_BARRIER(MPI_COMM_WORLD,info)

         endif
#endif

 1000    continue
C ---    LOOP over ports ends here


C ---    ===================== Residuals section ======================

C        Send Residuals: 1+NSCU solved equations, icomat materials
C        residuals are equal on all nodes if parallel

#ifdef PARALLEL_STAR
         if (MYID.eq.1) then
#endif
            IMAX=9+NSCU
            INUM=IMAX*icomat
            DO 191 j=1,IMAX
               DO 191 k=1,icomat
 191              udata((j-1)*icomat+k,1)=RESOR(j,k)
            continue

            if (COSU1D('residual',INUM,udata(1,1)).ne.0) then
               write(*,*) 'PROBLEMS creating DO for residual'
            endif

            write(REALTI,'(g16.5)') TIME
            if (COATTR('residual','REALTIME',REALTI).ne.0) then
               write(*,*) 'PROBLEMS creating Attribute for residual'
            endif

#ifdef PARALLEL_STAR
         endif
#endif


C ---    ========= Residuals section =========

C        Covise-Module is in server mode, finish it

#ifdef PARALLEL_STAR 
         if (MYID.eq.1) then
#endif
            if (COFINI().ne.0) then
               write(*,*) '  *************************************'
               write(*,*) '**** FINISH call not succeeded: exit ****'
               write(*,*) '  *************************************'
               open(unit=99,file="ABORT")
               write(99,*) "Abort"
               close(99)
            endif
#ifdef PARALLEL_STAR 
         endif
#endif


      endif

C ======================================================================
C ==== Now we might need some new BC : 
C                Do it here because POSDAT is called on all Procs on MPP
C ======================================================================

C --- not gotten or BC for this step: wait for it
C     MPP: switches inside COBREC()

         if (ITER.ge.ICOBOC) then

            if (COVERB().GT.0) then
               write(*,*) 'get BOCO: ITER=',ITER,'ICOBOC=',ICOBOC
            endif
            call COBREC()

         endif
      RETURN
      END



C*************************************************************************
C*************************************************************************
C*************************************************************************
C*************************************************************************
C********                                                         ********
C********     #####          ######                               ********
C********    #     #   ####  #     #   ####   #    #  #    #      ********
C********    #        #    # #     #  #    #  #    #  ##   #      ********
C********    #        #    # ######   #    #  #    #  # #  #      ********
C********    #        #    # #     #  #    #  #    #  #  # #      ********
C********    #     #  #    # #     #  #    #  #    #  #   ##      ********
C********     #####    ####  ######    ####    ####   #    #      ********
C********                                                         ********
C*************************************************************************
C*************************************************************************
C*************************************************************************
C*************************************************************************

      SUBROUTINE COBOUN(IREGION,SCALAR,U,V,W,TE,ED,T,DEN,TURINT,PRESS)
C     Boundary conditions at inlets
C*************************************************************************
C--------------------------------------------------------------------------*
C     STAR RELEASE 3.050                                                   *
C--------------------------------------------------------------------------*
C --- declares Simlib, Covise common-block and sizes
      INCLUDE 'coSimLib.inc'
      INCLUDE 'covise.inc'
      INCLUDE 'usrdat.inc'
      
      DIMENSION SCALAR(50)
      LOGICAL TURINT
     
C --- for everything before connection we leave the defaults
      if (IFIRST.eq.1) return

C ========================================================================
C --- set BC : only if we are interacting on it
C     use backward mapping function
      i = iregno(IREGION)

      if (i.gt.0) then
      
         if (icovel(i).gt.0) then
            u=covvel(1,i)
            v=covvel(2,i)
            w=covvel(3,i)
         endif
         
         if (icot  (i).gt.0)  t   = covt  (i)
         if (icoden(i).gt.0)  den = covden(i)
         
         if (icotur(i).gt.0) then
            te     = covtu1(i)
            ed     = covtu2(i)
            turint = .false.
         elseif (icotur(i).lt.0) then
            te     = covtu1(i)
            ed     = covtu2(i)
            turint = .true.
         endif
         
         if (icop  (i).gt.0)  den = covp  (i)
C        scalars later...         
      
      endif

C IREG = region

      RETURN
      END


C*************************************************************************
C*************************************************************************
C*************************************************************************
C*************************************************************************

C ------------  SUB to receive a set of BOCOs      
      
      SUBROUTINE COBREC()
      INCLUDE 'coSimLib.inc'
      INCLUDE 'covise.inc'
      INCLUDE 'usrdat.inc'

#ifdef PARALLEL_STAR 
C     only Master node receives
      if (MYID.eq.1) then
#endif

         if (CORECV(icostp,IPARAM).ne.IPARAM) then
            write(*,*) ''
            write(*,*) '  *******************************************'
            write(*,*) '****    BC block not received : QUIT       ****'
            write(*,*) '  *******************************************'
            write(*,*) ''
            icostp = -999

         endif
         
C        QUIT called: either not received boco or module sent QUIT
         if (icostp.eq.-999) then
            open(unit=99,file="ABORT")
            write(99,*) "Abort"
            close(99)
            icostp=1
         endif

#ifdef PARALLEL_STAR 
      endif
      
C     distribute to all nodes
      call MPI_BCAST(icostp,IBCLIN,MPI_INTEGER,0,MPI_COMM_WORLD,INFO)
      call MPI_BCAST(covvel,IBCLFL,MPI_REAL   ,0,MPI_COMM_WORLD,INFO)
#endif

C     set new stop-step
      ICOBOC=ITER+ICOSTP

#ifdef VERBOSE

#ifdef PARALLEL_STAR 
C     only Master should write
      if (MYID.eq.1) then
#endif

      write(*,*) ''
      write(*,*) '  *******************************************'
      write(*,*) '****    BC block  received : next after',ICOBOC
      write(*,*) '  *******************************************'
      write(*,*) ''

#ifdef PARALLEL_STAR 
      endif
#endif

#endif

      RETURN
      END




