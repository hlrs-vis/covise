
C*************************************************************************
      SUBROUTINE BCDEFI(SCALAR,U,V,W,TE,ED,T,DEN,TURINT)
C     Boundary conditions at inlets
C*************************************************************************
C--------------------------------------------------------------------------*
C     STAR RELEASE 3.050                                                   *
C--------------------------------------------------------------------------*
      INCLUDE 'comdb.inc'

      COMMON/USR001/INTFLG(100)
      
      DIMENSION SCALAR(50)
      LOGICAL TURINT
      INCLUDE 'usrdat.inc'
      DIMENSION SCALC(50)
      EQUIVALENCE( UDAT12(001), ICTID )
      EQUIVALENCE( UDAT04(002), DENC )
      EQUIVALENCE( UDAT04(003), EDC )
      EQUIVALENCE( UDAT02(005), PR )
      EQUIVALENCE( UDAT04(005), PRC )
      EQUIVALENCE( UDAT04(009), SCALC(01) )
      EQUIVALENCE( UDAT04(007), TC )
      EQUIVALENCE( UDAT04(008), TEC )
      EQUIVALENCE( UDAT04(059), UC )
      EQUIVALENCE( UDAT04(060), VC )
      EQUIVALENCE( UDAT04(061), WC )
      EQUIVALENCE( UDAT04(064), UCL )
      EQUIVALENCE( UDAT04(065), VCL )
      EQUIVALENCE( UDAT04(066), WCL )
      EQUIVALENCE( UDAT02(070), X )
      EQUIVALENCE( UDAT02(071), Y )
      EQUIVALENCE( UDAT02(072), Z )

c     write(*,*) '+++++++ bcdefi ++++ ITER=',ITER,'  +++ region=',IREG
      call flush(6)

C --- This is OUR part for COVISE : Dummy is the pressure...
      call COBOUN(IREG,SCALAR,U,V,W,TE,ED,T,DEN,TURINT,DUMMY)

      RETURN
      END
C
    
    
    
    
      
C-----------------------------------------------------------------------
C
C    This subroutine enables the user to specify INLET boundary
C    conditions for U,V,W,TE,ED,T and SCALAR.
C
C     Set TURINT=.TRUE.   if turbulence intensity and length scale are
C                         specified as TE and ED respectively
C     Set TURINT=.FALSE.  if k and epsilon are specified as TE and
C                         ED respectively
C
C    ** Parameters to be returned to STAR: U,V,W,TE,ED,T,
C                                          SCALAR, DEN, TURINT
C
C    NB U,V and W are in the local coordinate-system of the
C    inlet boundary.
C
C-----------------------------------------------------------------------
C
C     Sample coding: To specify inlet values for region 1
C
C      IF(IREG.EQ.1) THEN
C        TURINT=.FALSE.
C        U=
C        V=
C        W=
C        TE=
C        ED=
C        T=
C        SCALAR(1)=
C        DEN=
C      ENDIF
C-------------------------------------------------------------------------

