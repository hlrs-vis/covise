
C*************************************************************************
      SUBROUTINE NEWXYZ(VCORN)
C     New X, Y and Z coordinates
C*************************************************************************
C--------------------------------------------------------------------------*
C     STAR RELEASE 3.050                                                   *
C--------------------------------------------------------------------------*
      INCLUDE 'comdb.inc'

      COMMON/USR001/INTFLG(100)

      DIMENSION VCORN(3,NDMAXU)
      INCLUDE 'usrdat.inc'
C-------------------------------------------------------------------------
C
C    This subroutine enables the user to calculate new Cartesian vertex
C    coordinates of the mesh x=VCORN(1,IV), y=VCORN(2,IV), z=VCORN(3,IV),
C    where IV is the vertex number.
C
C    ** Parameters to be returned to STAR:  VCORN
C
C-------------------------------------------------------------------------
C
C     Sample coding: Vertex movement calculation for a reciprocating
C                    piston in a fixed cylinder, where the piston is
C                    activated by a rotating crank mechanism, whose para-
C                    meters are:
C                              C = connecting rod length
C                              A = crank radius
C                              OMEGA = rotational speed
C                              DELTHE = crank angle covered per time step
C                              ZREF = z coordinate below which the mesh
C                                        remains unchanged
C-------------------------------------------------------------------------
C      PI=3.14159265
C      C=40.0
C      A=9.0
C      OMEGA=100.
C      ZREF=1.
C      DELTHE=PI*OMEGA/30.*DT
C      THETA=REAL(ITER)*DELTHE
C      TN=SQRT(C**2-A**2*(SIN(THETA-DELTHE)**2))
C      ALFN=A*(1.-COS(THETA-DELTHE))-C+TN
C      TNP1=SQRT(C**2-A**2*(SIN(THETA)**2))
C      ALFNP1=A*(1.-COS(THETA))-C+TNP1
C      DELALF=ALFNP1-ALFN
CC---- LOCATE MAX Z
C      ZMAX=VCORN(3,1)
C      DO 20 I=2,NDMAXU
C      IF(VCORN(3,I).GT.ZMAX) ZMAX=VCORN(3,I)
C   20 CONTINUE
CC---- CALCULATE NEW Z VALUES
C      DO 10 I=1,NDMAXU
C      IF(VCORN(3,I).LE.ZREF) GO TO 10
C      VCORN(3,I)=VCORN(3,I)-DELALF*(VCORN(3,I)-ZREF)/(ZMAX-ZREF)
C   10 CONTINUE
C-------------------------------------------------------------------------C
      RETURN
      END
C
