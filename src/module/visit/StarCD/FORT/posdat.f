
C*************************************************************************
      SUBROUTINE POSDAT(KEY,VOL,U,TE,ED,T,P,VIST,DEN,CP,VISM,CON,
     *  F,ICLMAP,ICTID,RESOR,VF,FORCB,IRN,PREFM,LEVEL)
C     Post-process data
C*************************************************************************
C--------------------------------------------------------------------------*
C     STAR RELEASE 3.050                                                   *
C--------------------------------------------------------------------------*
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
      INCLUDE 'usrdat.inc'
      
c     write(*,*) '+++++++ POSDAT ++++ ITER=',ITER,'  ++++ level=',LEVEL
      call FLUSH(6)

C --- Connect to covise if we aren't yet
      if (IFIRST.EQ.1) call COSTAR(ICLMAP,KEY)

C --- AFTER the required number of steps: call covise output
      
      if (level.EQ.2.and.ITER.ge.ICOBOC) then
        call COPOST(KEY,VOL,U,TE,ED,T,P,VIST,DEN,CP,VISM,CON,
     *              F,ICLMAP,ICTID,RESOR,VF,FORCB,IRN,PREFM,LEVEL,
     *              NCOUS1,COVUS1,NCOUS2,COVUS2)
      endif

      RETURN
      END
C
