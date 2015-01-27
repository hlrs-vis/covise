
C ----------------------------------------------------------------------------
C
C   File eurovis.f(date) contains the FORTRAN subroutines X2XC, (X2XF), SLINC
C   and INTEVC. The routines are defined below.  
C
C ----------------------------------------------------------------------------
C
C     AMENDMENT HISTORY:
C
C         26.03.93    Annotation of previous changes removed.
C                     Changes made to SLINC so that the tangecy condition
C                     (if requested) is only applied if otherwise the 
C                     streamline hits a solid wall. These changes are annotated
C                     *//* 26.03.93.
C
C  17.09, 28.09.92    Following changes made:
C                     X2XC:-
C                       Convergence factor changed from 0.00001 to 0.0001,
C	                CIRA addendum for IRTN = 2 removed,
C                       Test for 0 <= P,Q,R <= 1 inserted (previously done
C                       in QUAD).
C                     FACE:- 
C                       QUAD called with NROOT=0 (Exclude roots outside 0,1).  
C                     ITER:-
C                       Modified to deal with > 100 iterations,
C                       QUAD called with NROOT=1 (Include roots outside 0,1).
C                     QUAD:-
C                       NROOT parameter added to specify if roots outside range             
C                       0,1 are allowed.             
C                     TRIANGLE:-               
C                       Various corrections and enhancements.            
C                     SLINC:-               
C                       Recording of iclip data (inserted by CIRA) removed,             
C                       ISOLID changed from 1 to 3 to reflect K=1 for 2D surface,             
C                       Label 3 moved to include test of negative delta,             
C                       Step using velocity at start of step now clipped if step             
C                            ends outside cell,        
C                       Calculation of SPATH2 corrected to use NCS-2 (not NCS-1)             
C   
C         27.11.91    Following changes made to SLINC to correct bug in ITER:
C                     QUAD modified to only return roots in range (0,1).
C                     Checks for range (0,1) removed from routines calling
C                     quad.  
C
C         29.10.91    X2XC and INTEVC modified to allow for the one 
C                     dimensional case where NDXC=1. 
C
C         16.10.91    All routines modified to allow for additional parameters
C                     NDX and NDXC specifying the number of cartesian
C                     coordinates and cell coordinates respectively.
C
C                     New routine INTEVC (Interpolate variable in cell)
C                     provided to return the interpolated values of NDX
C                     variables at the point XC specified in cell coordinates.
C
C         04.09.91    First version of SLINC, X2XC and X2XF.
C
C -----------------------------------------------------------------------------
C 
C     X2XC in this version requires the number of cartesian coordinates
C     specified in the NDX paremeter to be 3. NDXC may be 1, 2 or 3.
C
C     X2XC called with the parameter NDXC = 3 (3 dimensions), checks 
C     whether the point X(i,j,k) lies within the cell GC and if so
C     sets IRTN = 0 and XC to identify the position of X within cell.
C     Otherwise IRTN = 1. If the parameter NDXC is 2, indicating a two
C     dimensional problem, X2XC calls X2XF. If the parameter NDXC is 1
C     the routine returns the foot of the perpendicular from X to the
C     line joining the 2 specified points.
C
C     X2XF finds the foot of the perpendicular from the point X(i,j,k) to
C     the face. If the iterative process fails to find the foot of the
C     perpendicular in 100 iterations then IRTN = 1, otherwise IRTN = 0.  
C
C     Given the values of a set of variables at the 4 or 8 vertices of a 
C     cell, INTEVC finds the interpolated values at a point specified in  
C     cell coordinates.  
C
C     SLINC traces particle path through cell.
C
C     Return values from SLINC are:
C	   IRTN = 0 Path calculated successfully
C	        = 1 Zero length path
C               = 2 NISMAX exceeded
C	        = 3 NPMAX exceeded
C	       	= 4 Tolerance specified in DELTA not achieved in 100 iterations
C	      	= 5 Point of zero velocity reached
C  		
C GIVEN THE ORDINATES AND VELOCITY COMPONENTS AT THE 8
C CORNERS OF A CELL, THE INITIAL POSITION OF THE PARTICLE
C AND THE LOCATION (IF ANY) OF SOLID SURFACES, FIND ITS PATH
C THROUGH THE CELL, ITS POINT OF EXIT  AND THE STEP LENGTH
C NEEDED TO RETAIN THE PRESCRIBED ACCURACY.  THE STEP
C LENGTH IS REPEATEDLY HALVED UNTIL THE EXIT POINTS
C FROM TWO SUCCESSIVE TRACES ARE CLOSE ENOUGH.
C
C  - GEOMETRY OF THE CELL - 
C
C        H7__________G8  
C        '|          '|
C    D5'_________C6'  |               R
C     |   |       |   |               ^      Q
C     |   |       |   |               |    '
C     |   |       |   |               |  '
C     |  E3_______|__F4               +'--->P
C     |  '        |  '   
C    A1'_________B2'    
C                                 
C    
C       - FACE -      - NUM -  
C                                  
C     1 - 5 - 7 - 3      1       P = 0      IMIN
C     2 - 6 - 8 - 4      2       P = 1      IMAX
C     1 - 2 - 6 - 5      3       Q = 0      JMIN
C     3 - 4 - 8 - 7      4       Q = 1      JMAX
C     1 - 2 - 4 - 3      5       R = 0      KMIN
C     5 - 6 - 8 - 7      6       R = 1      KMAX
C    
C ----------------------------------------------------------------------

C    

C     F(P,Q,R) = (1-P)(1-Q)(1-R)FA +    P (1-Q)(1-R)FB
C               +   P (1-Q)   R FC + (1-P)(1-Q)   R FD
C               +(1-P)   Q (1-R)FE +    P    Q (1-R)FF
C               +   P    Q    R FG + (1-P)   Q    R FH
C     where F=X,Y OR Z.
C

C ----------------------------------------------------------------------


* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
*    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

C ----------------------------------------------------------------------

          SUBROUTINE X2XC(NDX,NDXC,X,GC,XCELL,IRTN)

C ----------------------------------------------------------------------
C
C   Parameters:
C
C      NDX     input number of cartesian dimensions (must be 3)
C      NDXC    input number of computational dimensions
C      X       input vector containing point in cartesian coordinates
C      GC      input bidimensional array containing coordinates of cell nodes
C                  first index is cartesian coord. index (1:NDX)
C                  second index is node index (1:2**NDXC)
C      XCELL   output vector containing point in cell coordinates (1:NDXC)
C      IRTN    output return code  (=1 : point not in cell)

C CIRA addendum begin
C      IRTN    output return code  (=2 : stop iter | nface)
C CIRA addendum end              

C
C   Method:
C
C If NDXC equals 1 the foot of the perpendicular from X to the line defined
C by GC is found. 
C
C If NDXC equals 2 the routine X2XF is called.
C
C If NDXC equals 3 the routine first checks that it is possible for the point 
C X to lie in the cell GC, by checking that the cell is intersected by all 3 
C planes X=XBAR,Y=YBAR,Z=ZBAR. The procedure is then to find where the line 
C X=XBAR,Y=YBAR cuts the faces, and then where this line cuts the plane Z=ZBAR.
C
C FACE   - Finds where the line cuts a cell face.
C ITER   - Finds wher the line cuts Z=ZBAR.
C QUAD   - Solves simultaneous equns. for two of the parametric variables.
C
C THE FOLLOWING ASSUMPTIONS (ALL JUSTIFIABLE) ARE MADE:
C   1. SINCE EACH FACE IS DESCRIBED BY BI-LINEAR EXPRESSIONS THE LINE
C      CUTS IT 0,1 OR 2 TIMES
C   2. IF A FACE IS CUT ONCE BY THE LINE THEN ONE OTHER FACE OF THE CELL
C      IS ALSO CUT ONCE (IN ADDITION OTHER FACES MAY BE CUT TWICE). IF
C      THE REQUIRED POINT IS WITHIN THE CELL THEN IT WILL LIE ON THE
C      LINE BETWEEN THESE FACES
C   3. IF THE LINE CUTS A FACE TWICE IT MAY ALSO CUT OTHER FACES
C               
 

	DIMENSION X(NDX),GC(NDX,2**NDXC),XCELL(NDXC)
	
	COMMON/COM/XA,YA,ZA,XB,YB,ZB,XC,YC,ZC,XD,YD,ZD,
     1           XE,YE,ZE,XF,YF,ZF,XG,YG,ZG,XH,YH,ZH,
     2           XBAR,YBAR,ZBAR,
     3           S1,T1,S2,T2,NCUT,
     4           P1,Q1,R1,P2,Q2,R2,P,Q,R,IFIND,CVGE


      IF (NDX .NE. 3) STOP 'X2XC: NDX NOT EQUAL TO 3'

      IF (NDXC .EQ. 1) THEN
         TOP = 0.
         DENOM = 0.	
         DO I=1,3
            TOP = TOP + (GC(I,2)-GC(I,1))*(X(I)-GC(I,1))
            DENOM = DENOM + (GC(I,2)-GC(I,1))**2
         ENDDO

         IF (DENOM .EQ. 0) THEN
            XCELL(1) = 0.
         ELSE
            XCELL(1) = TOP/DENOM
         ENDIF

         IF (XCELL(1) .LT. 0. .OR. XCELL(1) .GT. 1.) THEN
            IRTN = 1
         ELSE
            IRTN = 0
         ENDIF
       
         RETURN

      ENDIF   		  

      IF (NDXC .EQ. 2) THEN

         CALL X2XF (X,GC,XCELL,IRTN)
         RETURN
      ENDIF
      
      IF (NDXC .NE. 3) STOP 'X2XC: INVALID NDXC'

      XBAR=X(1)
      YBAR=X(2)
      ZBAR=X(3)
      XA=GC(1,1)
      YA=GC(2,1)
      ZA=GC(3,1)      
      XB=GC(1,2)
      YB=GC(2,2)
      ZB=GC(3,2)      
      XC=GC(1,6)
      YC=GC(2,6)
      ZC=GC(3,6)      
      XD=GC(1,5)
      YD=GC(2,5)
      ZD=GC(3,5)      
      XE=GC(1,3)
      YE=GC(2,3)
      ZE=GC(3,3)      
      XF=GC(1,4)
      YF=GC(2,4)
      ZF=GC(3,4)      
      XG=GC(1,8)
      YG=GC(2,8)
      ZG=GC(3,8)      
      XH=GC(1,7)
      YH=GC(2,7)
      ZH=GC(3,7)      

C  Check whether X,Y or Z are outside the minimum/maximum range of
C  X,Y or Z for the cell. If so the point does not lie in the cell.
 
      XMI=AMIN1(XA,XB,XC,XD,XE,XF,XG,XH)
      XMA=AMAX1(XA,XB,XC,XD,XE,XF,XG,XH)
      IF((XBAR-XMI)*(XBAR-XMA).GT.0) GOTO 100

      YMI=AMIN1(YA,YB,YC,YD,YE,YF,YG,YH)
      YMA=AMAX1(YA,YB,YC,YD,YE,YF,YG,YH)
      IF((YBAR-YMI)*(YBAR-YMA).GT.0) GOTO 100

      ZMI=AMIN1(ZA,ZB,ZC,ZD,ZE,ZF,ZG,ZH)
      ZMA=AMAX1(ZA,ZB,ZC,ZD,ZE,ZF,ZG,ZH)
      IF((ZBAR-ZMI)*(ZBAR-ZMA).GT.0) GOTO 100

      CVGE=0.0001*(ZMA-ZMI)

      NFACE=0

C FACE ABCD

      CALL FACE(XA,YA,ZA,
     1          XB,YB,ZB,
     2          XC,YC,ZC,
     3          XD,YD,ZD)

      IF(NCUT.EQ.1)THEN
            NFACE=1
            P1=S1
            Q1=0.
            R1=T1
      ELSE IF(NCUT.EQ.2)THEN
            P1=S1
            Q1=0.
            R1=T1
            P2=S2
            Q2=0.
            R2=T2
            CALL ITER

            IF(IFIND.NE.0) GOTO 200
      ENDIF

C FACE BFGC

      CALL FACE(XB,YB,ZB,
     1          XF,YF,ZF,
     2          XG,YG,ZG,
     3          XC,YC,ZC)
 
      IF(NCUT.EQ.1)THEN
            IF(NFACE.EQ.0)THEN
                  NFACE=1
                  P1=1.
                  Q1=S1
                  R1=T1
            ELSE
                  NFACE=2
                  P2=1.
                  Q2=S1
                  R2=T1
                  CALL ITER

                  GOTO 200
            ENDIF
      ELSE IF(NCUT.EQ.2)THEN
            P1=1.
            Q1=S1
            R1=T1
            P2=1.
            Q2=S2
            R2=T2
            CALL ITER

            IF(IFIND.NE.0) GOTO 200
      ENDIF


C FACE EFGH

      CALL FACE(XE,YE,ZE,
     1          XF,YF,ZF,
     2          XG,YG,ZG,
     3          XH,YH,ZH)
      IF(NCUT.EQ.1)THEN
            IF(NFACE.EQ.0)THEN
                  NFACE=1
                  P1=S1
                  Q1=1.
                  R1=T1
            ELSE
                  NFACE=2
                  P2=S1
                  Q2=1.
                  R2=T1
                  CALL ITER

                  GOTO 200
            ENDIF
      ELSE IF(NCUT.EQ.2)THEN
            P1=S1
            Q1=1.
            R1=T1
            P2=S2
            Q2=1.
            R2=T2
            CALL ITER

            IF(IFIND.NE.0) GOTO 200
      ENDIF


C FACE AEHD

      CALL FACE(XA,YA,ZA,
     1          XE,YE,ZE,
     2          XH,YH,ZH,
     3          XD,YD,ZD)
 
      IF(NCUT.EQ.1)THEN
            IF(NFACE.EQ.0)THEN
                  NFACE=1
                  P1=0.
                  Q1=S1
                  R1=T1
            ELSE
                  NFACE=2
                  P2=0.
                  Q2=S1
                  R2=T1
                  CALL ITER

                  GOTO 200
            ENDIF
      ELSE IF(NCUT.EQ.2)THEN
            P1=0.
            Q1=S1
            R1=T1
            P2=0.
            Q2=S2
            R2=T2
            CALL ITER

            IF(IFIND.NE.0) GOTO 200
      ENDIF


C FACE ABFE

      CALL FACE(XA,YA,ZA,
     1          XB,YB,ZB,
     2          XF,YF,ZF,
     3          XE,YE,ZE)
      IF(NCUT.EQ.1)THEN
            IF(NFACE.EQ.0)THEN
                  NFACE=1
                  P1=S1
                  Q1=T1
                  R1=0.
            ELSE
                  NFACE=2
                  P2=S1
                  Q2=T1
                  R2=0.
                  CALL ITER

                  GOTO 200
            ENDIF
      ELSE IF(NCUT.EQ.2)THEN
            P1=S1
            Q1=T1
            R1=0.
            P2=S2
            Q2=T2
            R2=0.
            CALL ITER

            IF(IFIND.NE.0) GOTO 200
      ENDIF


C FACE DCGH

      CALL FACE(XD,YD,ZD,
     1          XC,YC,ZC,
     2          XG,YG,ZG,
     3          XH,YH,ZH)
      IF(NCUT.EQ.1)THEN
 
            if(nface.eq.0)then
               irtn = 2
               return
            endif

            NFACE=2
            P2=S1
            Q2=T1
            R2=1.
            CALL ITER

            GOTO 200
      ELSE IF(NCUT.EQ.2)THEN
            P1=S1
            Q1=T1
            R1=1.
            P2=S2
            Q2=T2
            R2=1.
            CALL ITER

            GOTO 200
      ENDIF
      IFIND=0
      GOTO 100

200   IF (IFIND.EQ.1 .AND.
     1    0.LE.P .AND. P.LE.1 .AND. 
     2    0.LE.Q .AND. Q.LE.1 .AND. 
     3    0.LE.R .AND. R.LE.1) THEN
            XCELL(1)= P
            XCELL(2)= Q
            XCELL(3)= R
            IRTN = 0
            RETURN

      ELSEIF (IFIND .EQ. 2) THEN
            IRTN = 2
            RETURN     	

      ENDIF

C  Point not in cell

100   CONTINUE

      IRTN = 1
      RETURN

      END


C ----------------------------------------------------------------------

C ----------------------------------------------------------------------
      SUBROUTINE FACE(XU,YU,ZU,
     1                XV,YV,ZV,
     2                XW,YW,ZW,
     3                XR,YR,ZR)
C ----------------------------------------------------------------------
C
C FIND WHERE THE LINE X=XBAR,Y=YBAR CUTS THE FACE. THE NO. OF TIMES
C IT CUTS IT IS PLACED IN NCUT AND VALUES OF S AND T AT THE POINT(S)
C ARE PLACED IN S1,T1 (AND S2,T2).
C
C    R __________ W
C     I          I
C    tI          I
C    ^I          I
C    |I          I
C    UI__________IV
C      -->s
C
C SOLVE THE FOLLOWING EQUATIONS FOR S AND T
C XBAR=(1-S)(1-T)XU+S(1-T)XV+(1-S)T.XR+ST.XW,OR AX.ST+BX.S+CX.T+DX=0
C YBAR=(1-S)(1-T)YU+S(1-T)YV+(1-S)T.YR+ST.YW,OR AY.ST+BY.S+CY.T+DY=0
C

      COMMON/COM/XA,YA,ZA,XB,YB,ZB,XC,YC,ZC,XD,YD,ZD,
     1           XE,YE,ZE,XF,YF,ZF,XG,YG,ZG,XH,YH,ZH,
     2           XBAR,YBAR,ZBAR,
     3           S1,T1,S2,T2,NCUT,
     4           P1,Q1,R1,P2,Q2,R2,P,Q,R,IFIND,CVGE


      NCUT=0
      XMIN=AMIN1(XU,XV,XW,XR)
      XMAX=AMAX1(XU,XV,XW,XR)
      IF((XMAX-XBAR)*(XMIN-XBAR).GT.0)RETURN
      YMIN=AMIN1(YU,YV,YW,YR)
      YMAX=AMAX1(YU,YV,YW,YR)
      IF((YMAX-YBAR)*(YMIN-YBAR).GT.0)RETURN

      AX=XU-XV-XR+XW
      BX=XV-XU
      CX=XR-XU
      DX=XU-XBAR
      AY=YU-YV-YR+YW
      BY=YV-YU
      CY=YR-YU
      DY=YU-YBAR

      CALL QUAD(AX,BX,CX,DX,AY,BY,CY,DY,NCUT,S1,T1,S2,T2,0)

      RETURN
      END
 
C ----------------------------------------------------------------------

C ----------------------------------------------------------------------
      SUBROUTINE ITER
C ----------------------------------------------------------------------
C
C GIVEN TWO POINTS 1,2 ON THE LINE X=XBAR,Y=YBAR FIND WHERE THE LINE
C MEETS THE PLANE Z=ZBAR, IF WITHIN THE CELL. IF SUCCESSFUL SET IFIND.
C
      COMMON/COM/XA,YA,ZA,XB,YB,ZB,XC,YC,ZC,XD,YD,ZD,
     1           XE,YE,ZE,XF,YF,ZF,XG,YG,ZG,XH,YH,ZH,
     2           XBAR,YBAR,ZBAR,
     3           S1,T1,S2,T2,NCUT,
     4           P1,Q1,R1,P2,Q2,R2,P,Q,R,IFIND,CVGE
C
      ZLIN(P,Q,R)=(1-P)*(1-Q)*(1-R)*ZA+(  P)*(1-Q)*(1-R)*ZB
     1           +(  P)*(1-Q)*(  R)*ZC+(1-P)*(1-Q)*(  R)*ZD
     2           +(1-P)*(  Q)*(1-R)*ZE+(  P)*(  Q)*(1-R)*ZF
     3           +(  P)*(  Q)*(  R)*ZG+(1-P)*(  Q)*(  R)*ZH
C

      P=P1
      Q=Q1
      R=R1

      IFIND=0
      Z1=ZLIN(P1,Q1,R1)
      Z2=ZLIN(P2,Q2,R2)

      IF (Z1 .EQ. ZBAR) THEN
         IFIND=1
         RETURN
      ENDIF
   
      IF (Z2 .EQ. ZBAR) THEN
         P=P2
         Q=Q2
         R=R2
         IFIND=1
         RETURN
      ENDIF

      IF((Z1-ZBAR)*(Z2-ZBAR).GT.0) RETURN

C
C CHOOSE WHICH OF P,Q OR R TO USE WHEN ITERATING ON Z
C
      IT=0
      P0=ABS(P1-P2)
      Q0=ABS(Q1-Q2)
      R0=ABS(R1-R2)
      IF(R0.GT.P0.AND.R0.GT.Q0) GOTO 3
      IF(Q0.GT.P0.AND.Q0.GT.R0) GOTO 2

C    --- ITERATE ON Z AND P ---

1     P=P1+(ZBAR-Z1)*(P2-P1)/(Z2-Z1)

C        FIND R,Q FROM
C                AX.RQ+BX.R+CX.Q+DX=0
C                AY.RQ+BY.R+CY.Q+DY=0

      AX=(1-P)*(XA-XD-XE+XH)
     &      +P*(XB-XC-XF+XG)
      BX=(1-P)*(XD-XA)
     &      +P*(XC-XB)
      CX=(1-P)*(XE-XA)
     &      +P*(XF-XB)
      DX=(1-P)*XA
     &      +P*XB-XBAR
      AY=(1-P)*(YA-YD-YE+YH)
     &      +P*(YB-YC-YF+YG)
      BY=(1-P)*(YD-YA)
     &      +P*(YC-YB)
      CY=(1-P)*(YE-YA)
     &      +P*(YF-YB)
      DY=(1-P)*YA
     &      +P*YB-YBAR

      CALL QUAD(AX,BX,CX,DX,AY,BY,CY,DY,NCUT,R0,Q0,R,Q,1)

      IF(NCUT.EQ.0) then
         ifind = 2
         return
      endif

      Z0=ZLIN(P,Q0,R0)
      IF (NCUT.EQ.1) THEN
         R=R0
         Q=Q0
         Z=Z0
      ELSE
         Z=ZLIN(P,Q,R)
         IF((Z1-Z)*(Z2-Z).GT.0)THEN
            R=R0
            Q=Q0
            Z=Z0
         ELSE IF((Z1-Z0)*(Z2-Z0).LE.0)THEN
            IF(ABS(ZBAR-Z0).LT.ABS(ZBAR-Z))THEN
                  R=R0
                  Q=Q0
                  Z=Z0
            ENDIF
         ENDIF
      ENDIF
      IF(ABS(Z-ZBAR).LT.CVGE)THEN
            IF(0.LE.P.AND.P.LE.1)IFIND=1
            RETURN
      ENDIF
      IT=IT+1

      IF(IT.GT.100) THEN
         IF(0.LE.P .AND. P.LE.1) THEN
            IFIND = 1 
         ELSE
            IFIND = 2
         ENDIF
    
         RETURN
      ENDIF   

      IF((ZBAR-Z1)*(ZBAR-Z).GT.0)THEN
            Z1=Z
            P1=P
      ELSE
            Z2=Z
            P2=P
      ENDIF

      GOTO 1


C   --- ITERATE ON Z AND Q ---

2     Q=Q1+(ZBAR-Z1)*(Q2-Q1)/(Z2-Z1)

C            FIND P,R FROM
C                  AX.PR+BX.P+CX.R+DX=0
C                   AY.PR+BY.P+CY.R+DY=0

      AX=(1-Q)*(XA-XB-XD+XC)
     &      +Q*(XE-XF-XH+XG)
      BX=(1-Q)*(XB-XA)
     &      +Q*(XF-XE)
      CX=(1-Q)*(XD-XA)
     &      +Q*(XH-XE)
      DX=(1-Q)*XA
     &      +Q*XE-XBAR
      AY=(1-Q)*(YA-YB-YD+YC)
     &      +Q*(YE-YF-YH+YG)
      BY=(1-Q)*(YB-YA)
     &      +Q*(YF-YE)
      CY=(1-Q)*(YD-YA)
     &      +Q*(YH-YE)
      DY=(1-Q)*YA
     &      +Q*YE-YBAR

      CALL QUAD(AX,BX,CX,DX,AY,BY,CY,DY,NCUT,P0,R0,P,R,1)

      IF(NCUT.EQ.0) then
         ifind = 2
         return
      endif

      Z0=ZLIN(P0,Q,R0)
      IF (NCUT.EQ.1) THEN
         P=P0
         R=R0
         Z=Z0
      ELSE
         Z=ZLIN(P,Q,R)
         IF((Z1-Z)*(Z2-Z).GT.0)THEN
            P=P0
            R=R0
            Z=Z0
         ELSE IF((Z1-Z0)*(Z2-Z0).LE.0)THEN
            IF(ABS(ZBAR-Z0).LT.ABS(ZBAR-Z))THEN
                  P=P0
                  R=R0
                  Z=Z0
            ENDIF
         ENDIF
      ENDIF
      IF(ABS(Z-ZBAR).LT.CVGE)THEN
            IF(0.LE.Q.AND.Q.LE.1.)IFIND=1
            RETURN
      ENDIF
      IT=IT+1

      IF(IT.GT.100) THEN
         IF(0.LE.Q .AND. Q.LE.1) THEN
            IFIND = 1 
         ELSE
            IFIND = 2
         ENDIF
    
         RETURN
      ENDIF   

      IF((ZBAR-Z1)*(ZBAR-Z).GT.0)THEN
            Z1=Z
            Q1=Q
      ELSE
            Z2=Z
            Q2=Q
      ENDIF

      GOTO 2


C    --- ITERATE ON Z AND R ---

3     R=R1+(ZBAR-Z1)*(R2-R1)/(Z2-Z1)

C            FIND P,Q FROM
C                    AX.PQ+BX.P+CX.Q+DX=0
C                    AY.PQ+BY.P+CY.Q+DY=0

      AX=(1-R)*(XA-XB-XE+XF)
     &      +R*(XD-XC-XH+XG)
      BX=(1-R)*(XB-XA)
     &      +R*(XC-XD)
      CX=(1-R)*(XE-XA)
     &      +R*(XH-XD)
      DX=(1-R)*XA
     &      +R*XD-XBAR
      AY=(1-R)*(YA-YB-YE+YF)
     &      +R*(YD-YC-YH+YG)
      BY=(1-R)*(YB-YA)
     &      +R*(YC-YD)
      CY=(1-R)*(YE-YA)
     &      +R*(YH-YD)
      DY=(1-R)*YA
     &      +R*YD-YBAR

      CALL QUAD(AX,BX,CX,DX,AY,BY,CY,DY,NCUT,P0,Q0,P,Q,1)

      IF(NCUT.EQ.0) then
         ifind = 2
         return
      endif

      Z0=ZLIN(P0,Q0,R)
      IF (NCUT.EQ.1) THEN
         P=P0
         Q=Q0
         Z=Z0
      ELSE
         Z=ZLIN(P,Q,R)
         IF((Z1-Z)*(Z2-Z).GT.0)THEN
            P=P0
            Q=Q0
            Z=Z0
         ELSE IF((Z1-Z0)*(Z2-Z0).LE.0)THEN
            IF(ABS(ZBAR-Z0).LT.ABS(ZBAR-Z))THEN
                  P=P0
                  Q=Q0
                  Z=Z0
            ENDIF
         ENDIF
      ENDIF
      IF(ABS(Z-ZBAR).LT.CVGE)THEN
            IF(0.LE.R.AND.R.LE.1)IFIND=1
            RETURN
      ENDIF
      IT=IT+1
      IF(IT.GT.100) THEN
         IF(0.LE.R .AND. R.LE.1) THEN
            IFIND = 1 
         ELSE
            IFIND = 2
         ENDIF
    
         RETURN
      ENDIF   

      IF((ZBAR-Z1)*(ZBAR-Z).GT.0)THEN
            Z1=Z
            R1=R
      ELSE
            Z2=Z
            R2=R
      ENDIF
      GOTO 3

      END
 
C ----------------------------------------------------------------------

C ----------------------------------------------------------------------
      SUBROUTINE QUAD(A1,B1,C1,D1,A2,B2,C2,D2,N,X1,Y1,X2,Y2,NROOT)
C ----------------------------------------------------------------------
C
C FINDS REAL ROOTS OF THE EQUS.
C     A1.XY+B1.X+C1.Y+D1=0
C     A2.XY+B2.X+C2.Y+D2=0
C
C PUTS ROOTS INTO (X1,Y1),(X2,Y2) AND THE NO. OF ROOTS INTO N
C IF NROOT=0 ROOTS OUTSIDE (0,1) ARE EXCLUDED	

      PARAMETER (EPSYLON=0.00001) 	

      N=0

      IF (A1.EQ.0.AND.B1.EQ.0) THEN
         IF (C1.EQ.0) RETURN
         Y1 = -D1/C1
         EX1 = A2*Y1+B2
         IF (EX1.EQ.0) RETURN
         N=1
         X1 = -(C2*Y1+D2)/EX1
         GOTO 100
      ENDIF

      IF (A2.EQ.0.AND.B2.EQ.0) THEN
         IF (C2.EQ.0) RETURN
         Y1 = -D2/C2
         EX1 = A1*Y1+B1
         IF (EX1.EQ.0) RETURN
         N=1
         X1 = -(C1*Y1+D1)/EX1
         GOTO 100
      ENDIF

      A=A2*C1-A1*C2
      B=A2*D1+B2*C1-A1*D2-B1*C2
      C=B2*D1-B1*D2

      IF (A.EQ.0) THEN
         IF (B.EQ.0) RETURN
         Y1 = -C/B
         EX1 = A1*Y1+B1
         EX2 = A2*Y1+B2
         IF (EX1.EQ.0.AND.EX2.EQ.0) RETURN
         N = 1
         IF (ABS(EX1).GT.ABS(EX2)) THEN
            X1 = -(C1*Y1+D1)/EX1
         ELSE
            X1 = -(C2*Y1+D2)/EX2
         ENDIF
         GOTO 100
      ENDIF

      EX=B*B-4*A*C
      IF(EX.LT.0)RETURN
      N=2
 
      EX=SQRT(EX)

****************************
*   To avoid problems when B**2 >> 4AC.
*   If B is positive, then the first Y root (Y1) is taken from 
*   Y = -B-sqrt(B**2 - 4AC) and if B negative, from Y=-B+sqrt(...).
*   The second root is calculated from Y1*Y2=C/A.
*
****************************

      IF (B.GT.0) THEN
         Y1 = (-B-EX)/(2*A)
      ELSE
         Y1 = (-B+EX)/(2*A)
      ENDIF

      Y2 = C/(A*Y1)

      EX1 = A1*Y1+B1
      EX2 = A1*Y2+B1
 
      IF (EX1*EX2.NE.0) THEN
         X1 = -(C1*Y1+D1)/EX1
         X2 = -(C1*Y2+D1)/EX2
         GOTO 200
      ENDIF
 
      IF (EX2.EQ.0) THEN
         EX2 = A2*Y2+B2
         IF (EX2.EQ.0) THEN
            N = 1
         ELSE
            X2 = -(C2*Y2+D2)/EX2
         ENDIF
      ELSE
         X2 = -(C1*Y2+D1)/EX2
      ENDIF

      IF (EX1.EQ.0) THEN
         EX1 = A2*Y1+B2
         IF (EX1.EQ.0) THEN
            N = N - 1
            IF (N.EQ.0) RETURN
            X1 = X2
            Y1 = Y2
         ELSE
            X1 = -(C2*Y1+D2)/EX1
         ENDIF
      ELSE
         X1 = -(C1*Y1+D1)/EX1
      ENDIF 

C   Case N=1 or 2:

200   IF(NROOT .NE. 0) RETURN
      IF(N.EQ.2.AND.(X2.LT.0.OR.X2.GT.1.OR.Y2.LT.0.OR.Y2.GT.1)) N=1

      IF(N.EQ.2.AND.(X1.LT.0.OR.X1.GT.1.OR.Y1.LT.0.OR.Y1.GT.1)) THEN
          N = 1
         X1 = X2           
         Y1 = Y2
         RETURN
      ENDIF

C    Case n=1:

100   IF(NROOT .NE. 0) RETURN
      IF (X1.LT.-EPSYLON .OR. X1.GT.1+EPSYLON .OR. 
     &              Y1.LT.-EPSYLON .OR. Y1.GT.1+EPSYLON) THEN
         N=0
      ELSE
         IF (X1.LT.0) X1=0      
         IF (X1.GT.1) X1=1      
         IF (Y1.LT.0) Y1=0      
         IF (Y1.GT.1) Y1=1
      ENDIF
      
      RETURN
      END
 
C ----------------------------------------------------------------------


* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
*    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

C ----------------------------------------------------------------------

      SUBROUTINE X2XF(FX,GF,XF,IRTN)            

C ----------------------------------------------------------------------
C
C Given a point FX = X0,Y0,Z0 find the foot of the perpendicular to the face
C GF. Return the face coordinates in XF. The routine uses an iterative 
C process and if this fails to converge in 100 iterations IRTN is set to 1, 
C otherwise IRTN = 0. 
C 
C ----------------------------------------------------------------------
C
C   For the face ABCD the coordinates of any                   D--------C
C   point are:                                                 |        |
C                                                           q^ |        |
C   X = (1-p)(1-q) XA + p(1-q) XB + pq XC (1-p)q XD          | |        |
C   Y = (1-p)(1-q) YA + p(1-q) YB + pq YC (1-p)q YD            A--------B
C   Z = (1-p)(1-q) ZA + p(1-q) ZB + pq ZC (1-p)q ZD              -->p
C
C   Distance of point (X0,Y0,Z0) from surface is given by
C
C             d^2 = (X-X0)^2 + (Y-Y0)^2 + (Z-Z0)^2
C
C   If (X,Y,Z) is on the normal to surface through (X0,Y0,Z0) then d^2 has
C   a minimum for this point. i.e.
C
C       (X-X0)Xp + (Y-Y0)Yp + (Z-Z0)Zp = 0
C       (X-X0)Xq + (Y-Y0)Yq + (Z-Z0)Zq = 0
C
C   where Xp = @x/@p etc.  and @ is delta.
C
C   To find p,q at the foot of the normal solve equns above for p,q using
C   Newton iteration.
C
C   Write f1(p,q) = (X-X0)Xp + (Y-Y0)Yp +(Z-Z0)Zp    
C         f2(p,q) = (X-X0)Xq + (Y-Y0)Yq +(Z-Z0)Zq    
C
C   If an approximation to p,q is p1,q1, then a better approximation is
C   p1+@p; q1+@q  where f1(p1+@p,q1+@q)=0 and f2(p1+@p,q1+@q)=0.
C
C   Approximating these 2 equations by
C
C           f1(p1,q1) + @p.@f1/@p + @q.@f1/@q = 0 
C           f2(p1,q1) + @p.@f2/@p + @q.@f2/@q = 0 
C
C   gives
C
C          @p =  (f2.f1q -f1.f2q) / (f1p.f2q -f2p.f1q)
C          @q = -(f2.f1p -f1.f2p) / (f1p.f2q -f2p.f1q)
C
C   and hence a better approximation to p,q.
C
C    Xp = (1-q)(XB-XA) +q (Xc-XD),   Xq = (1-p)(XD-XA) + p(XC-XB)
C    Yp = (1-q)(YB-YA) +q (Yc-YD),   Yq = (1-p)(YD-YA) + p(YC-YB)
C    Zp = (1-q)(ZB-ZA) +q (Zc-ZD),   Zq = (1-p)(ZD-ZA) + p(ZC-ZB)
C
C   Xpp = Xqq = 0;  Xpq = XC - XD - XB + XA  
C   Ypp = Yqq = 0;  Ypq = YC - YD - YB + YA  
C   Zpp = Zqq = 0;  Zpq = ZC - ZD - ZB + ZA  
C
C   f1p = (X-X0)Xpp + (Y-Y0)Ypp + (Z-Z0)Zpp + Xp^2 + Yp^2 + Zp^2  
C       = Xp^2 + Yp^2 + Zp^2  
C
C   f1q = (X-X0)Xpq + (Y-Y0)Ypq + (Z-Z0)Zpq + XpXq + YpYq + ZpZq
C   f2p = (X-X0)Xpq + (Y-Y0)Ypq + (Z-Z0)Zpq + XpXq + YpYq + ZpZq = f1q
C   
C   f2q = Xq^2 + Yq^2 + Zq^2
C
C ----------------------------------------------------------------------
  
      DIMENSION FX(3),GF(3,4),XF(2),XYZ(3),XX(3)

C   Try triangle ABD for initial approximation

      CALL TRIANGLE(FX,GF,1,2,4,3,P,Q,1,JRTN)

      IF (JRTN .EQ. 1) THEN

C   Point not found in ABD so use triangle CDB

         CALL TRIANGLE(FX,GF,3,4,2,1,P,Q,2,JRTN)

         P=1-P
         Q=1-Q

      ENDIF


2     D2LAST = 1.0E30
      ITER=0

1     ITER=ITER+1
      IF (ITER .GT. 100) THEN
         IRTN=1
         RETURN
      ENDIF 

C   Calculate coordinates of point in face.

      DO J=1,3

         XYZ(J) = (1-P)*(1-Q) * GF(J,1) + P*(1-Q) * GF(J,2)
     &           +        P*Q * GF(J,3) + (1-P)*Q * GF(J,4) 

      ENDDO

      F1 = 0
      F2 = 0
      F1P= 0
      F2Q= 0
      F1Q= 0

      DO J=1,3
         XP = (1-Q)*(GF(J,2)-GF(J,1)) + Q*(GF(J,3)-GF(J,4))
         XQ = (1-P)*(GF(J,4)-GF(J,1)) + P*(GF(J,3)-GF(J,2))
         XPQ = GF(J,3)-GF(J,4)-GF(J,2)+GF(J,1)
         F1 = F1 + (XYZ(J)-FX(J)) * XP
         F2 = F2 + (XYZ(J)-FX(J)) * XQ
         F1P = F1P + XP*XP
         F2Q = F2Q + XQ*XQ 
         F1Q = F1Q + (XYZ(J)-FX(J))*XPQ + XP*XQ
      ENDDO

      DENOM = F1P*F2Q - F1Q*F1Q

      DP = (F2*F1Q - F1*F2Q) / DENOM
      DQ = (F1*F1Q - F2*F1P) / DENOM

      D2 = DP**2 + DQ**2

      IF (D2 .GE. D2LAST .AND. D2 .LT. 0.0001) THEN
          XF(1)=P
          XF(2)=Q
          IRTN =0
          RETURN
      ENDIF

      D2LAST = D2

      P = P + DP
      Q = Q + DQ

      GOTO 1

      END

C ----------------------------------------------------------------------

      SUBROUTINE TRIANGLE(FX,GF,J1,J2,J3,J4,P,Q,ITRI,JRTN)            

C ----------------------------------------------------------------------

C   The subroutine TRIANGLE first finds the foot of the perpendicular from
C   the point FX to the triangle defined by the points J1,J2,J3 in GF. This
C   is the point P0,Q0.
C
C   If ITRI equals 1 (used for the first triangle in the quadrilateral) then
C   if the foot of the perpendicular is outside the triangle the routine 
C   returns failure with JRTN = 1.
C   
C   Otherwise the values of P0,Q0 are transformed from the triangular 
C   coordinates to face coordinates by solving the equations
C
C    (XB-XA)P0 + (XD-XA)Q0 = (XB-XA)P + (XD-XA)Q + (XA-XB+XC-XD)PQ
C    (YB-YA)P0 + (YD-YA)Q0 = (YB-YA)P + (YD-YA)Q + (YA-YB+YC-YD)PQ
C
C   where X and Y are selected from X,Y and Z as the two with the largest 
C   spread of values at the four vertices.

C ----------------------------------------------------------------------
  
      DIMENSION FX(3),GF(3,4),XF(2),XYZ(3),XM(3)      

      AB=0
      A12=0
      A13=0

      AD=0
      A23=0
  
      DO I=1,3

         A = GF(I,J2)-GF(I,J1)
         B = GF(I,J3)-GF(I,J1)
         
         AB = AB + A*A

         A12 = A12 + A*B
         A13 = A13 + A*(GF(I,J1)-FX(I))

         AD = AD + B*B

         A23 = A23 + B*(GF(I,J1)-FX(I))

      ENDDO

      DENOM = A12*A12 - AB*AD

C    Degenarate triangle gives P=Q=0.5

      IF (DENOM .EQ. 0) THEN
         P = 0.5
         Q = 0.5
         JRTN = 0 
         RETURN
      ENDIF
 
      Q0 = (AB*A23 - A12*A13) / DENOM

      P0 = (AD*A13 - A12*A23) / DENOM

      IF (ITRI.EQ.1 .AND. (P0.LT.0 .OR. Q0.LT.0 .OR. P0+Q0.GT.1) )  THEN
         JRTN = 1
         RETURN
      ENDIF
      

C   Find which of X,Y or Z has the least variation and set L1,L2 to identify 
C   the two with the most.

      DO I=1,3
         XMID=0
         DO J = 1,4
            XMID = XMID + GF(I,J)
         ENDDO
         XMID = XMID/4

         XM(I)= MAX(ABS(XMID-GF(I,J1)),ABS(XMID-GF(I,J2)),
     &              ABS(XMID-GF(I,J3)),ABS(XMID-GF(I,J4)))

      ENDDO

      L1 = 1
      L2 = 2

      XMIN = MIN(XM(1),XM(2),XM(3))
      IF (XMIN .EQ. XM(1)) THEN
         L1 = 3
      ELSEIF (XMIN .EQ. XM(2)) THEN
         L2 = 3
      ENDIF

C  Call QUAD to solve the following equations for P and Q:
C
C    (XB-XA)P0 + (XD-XA)Q0 = (XB-XA)P + (XD-XA)Q + (XA-XB+XC-XD)PQ
C    (YB-YA)P0 + (YD-YA)Q0 = (YB-YA)P + (YD-YA)Q + (YA-YB+YC-YD)PQ
C
C   Where X and Y are the pair from X,Y,Z that have the biggest spread.

      A1 = GF(L1,J1) - GF(L1,J2) + GF(L1,J3) - GF(L1,J4)
      B1 = GF(L1,J2) - GF(L1,J1)
      C1 = GF(L1,J4) - GF(L1,J1)
      D1 =   -B1*P0 - C1*Q0   

      A2 = GF(L2,J1) - GF(L2,J2) + GF(L2,J3) - GF(L2,J4)
      B2 = GF(L2,J2) - GF(L2,J1)
      C2 = GF(L2,J4) - GF(L2,J1)
      D2 =   -B2*P0 - C2*Q0   

      CALL QUAD(A1,B1,C1,D1,A2,B2,C2,D2,N,P1,Q1,P2,Q2,0)

      P = P1
      Q = Q1
      JRTN = 0

      IF (N .EQ. 2) THEN 

C  Take point nearest to P0,Q0

         D1=(P1-P0)**2+(Q1-Q0)**2
         D2=(P2-P0)**2+(Q2-Q0)**2
         IF (D1 .GT. D2) THEN
            P = P2
            Q = Q2
         ENDIF

      ELSEIF (N .EQ. 0) THEN
         
         P = P0
         Q = Q0
 
      ENDIF
 
      RETURN

      END

C ----------------------------------------------------------------------


* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
*    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 


C ----------------------------------------------------------------------

      SUBROUTINE SLINC (NDX,NDXC,GCI,VELCI,IBOCC,XC1,ISTEP,DL,DL1,
     &                  NPMAX,DELTA,NISMAX,NP,XP,XCP,
     &                  XC2,DL2,IRTN)

C ------------------------------------------------------------------------
C
C GIVEN THE ORDINATES AND VELOCITY COMPONENTS AT THE 8
C CORNERS OF A CELL, THE INITIAL POSITION OF THE PARTICLE
C AND THE LOCATION (IF ANY) OF SOLID SURFACES, FIND ITS PATH
C THROUGH THE CELL, ITS POINT OF EXIT  AND THE STEP LENGTH
C NEEDED TO RETAIN THE PRESCRIBED ACCURACY.  THE STEP
C LENGTH IS REPEATEDLY HALVED UNTIL THE EXIT POINTS
C FROM TWO SUCCESSIVE TRACES ARE CLOSE ENOUGH.
C
C  - GEOMETRY OF THE CELL - 
C
C         7___________8  
C        '|          '|
C     5'__________6'  |               R
C     |   |       |   |               ^      Q
C     |   |       |   |               |    '
C     |   |       |   |               |  '
C     |   3_______|___4               +'--->P
C     |  '        |  '  
C     1'__________2'    
C                               
C    
C       - FACE -      - NUM -  
C                                  
C     1 - 5 - 7 - 3      1       P = 0      IMIN
C     2 - 6 - 8 - 4      2       P = 1      IMAX
C     1 - 2 - 6 - 5      3       Q = 0      JMIN
C     3 - 4 - 8 - 7      4       Q = 1      JMAX
C     1 - 2 - 4 - 3      5       R = 0      KMIN
C     5 - 6 - 8 - 7      6       R = 1      KMAX
C    
C ----------------------------------------------------------------------
C
C TO FIND VELOCITY COMPONENTS AT ANY POINT, U,V,W 
C ARE ASSUMED TO VARY TRI - LINEARLY THROUGHOUT THE
C CELL. 
C
C TRI - LINEAR VARIATION BEING DEFINED AS:
C
C  F=(1 - p)(1 - q)(1 - r)Fi,j,k     + p(1 - q)(1 - r)Fi+1,j,k    +
C    (1 - p)(1 - q)r      Fi,j,k+1   + p(1 - q)r      Fi+1,j,k+1  +
C    (1 - p)q(1 - r)      Fi,j+1,k   + pq(1 - r)      Fi+1,j+1,k  +
C    (1 - p)qr            Fi,j+1,k+1 + pqr            Fi+1,j+1,k+1
C
C IF TANGENCY HAS BEEN REQUESTED THEN DP, DQ OR DR IS MODIFIED IN
C THE CELL NEAREST THE SURFACE BY MULTIPLYING IT BY P, Q OR R, OR
C 1-P, 1-Q OR 1-R AS APPROPRIATE.
C
C --------------------------------------------------------------------------
C
C   PARAMETERS:
C
C   NDX    - Number of cartesian dimensions (1 for X coord. only, ...) [Input]
C
C   NDXC   - Number of computational dimensions (1 for i only, ...) [Input]
C
C   GCI    - Geometry coords at cell nodes (xyz,node) [Input]
C
C   VELCI  - Velocity components at cell nodes (uvw,node) [Input]
C
C   IBOCC  - Boundary condition at cell faces  (face) [Input]
C                 =0: no condition, =1: tangency.
C
C   XC1    - Cell coords of starting point (pqr) [Input]
C
C   ISTEP  - Step type [Input]
C                 =1: equal distance,  =2: equal time, negative: upstream
C
C   DL     - Step length [Input]
C
C   DL1    - Offset of start point from DL [Input]
C              (First output point at length DL-DL1)
C
C   NPMAX  - Maximum number of path points [Input]
C                (gives max streamline length)
C
C   DELTA  - Tolerance on angular variation (radians) [Input]
C                If not positive: no check required
C
C   NISMAX - Maximum number of computational steps [Input]
C
C   NP     - Number of path points [Output]
C
C   XP     - Cartesian coords of path points (xyz,n) [Output]
C
C   XCP    - Cell coords of path points (pqr,n) [Output]
C
C   XC2    - Cell coords of ending point (pqr) [Output] 
C
C   DL2    - Length of final step [Output]
C
C   IRTN   - Return code [Output]
C	           = 0 : Path calculated successfully
C	           = 1 : Zero length path (path is external)
C                  = 2 : NISMAX exceeded
C	           = 3 : NPMAX exceeded
C	       	   = 4 : Tolerance specified in DELTA not achieved in 
C                      100 iterations
C	      	   = 5 : Point of zero velocity reached
C  		
C -------------------------------------------------------------------------
 
      DIMENSION XC1(NDX),XC2(NDX),GCI(NDX,2**NDXC),VELCI(NDX,2**NDXC)
      DIMENSION XP(NDX,NPMAX),XCP(NDX,NPMAX),IBOCC(2*NDXC)

C ---------------------------------------------------------------------------
C
C  WORKING VARIABLES
C
C   GC          - GCI expanded to (3,8)
C
C   VELC        - VELCI expanded to (3,8) 
C
C   P,Q,R       - Working values of cell coordinates (0 <= P,Q,R <= 1).
C
C   P0, Q0, R0  -  Starting values of P,Q,R for each step.
C
C   DP, DQ, DR  -  Increments in P, Q, R.  
C
C   NOUT        -  =0 unless face of cell reached, when NOUT=1
C  
C   NFLOP       - Flip-flop = 1 or 2 identifying storage for current point.
C
C   XPT(3,2)    - Current values of X,Y,Z in XPT(1-3,NFLOP) and values at
C                 previous step in XPT(1-3,3-NFLOP).
C
C   XLAST(3)    - Point on cell face where last iteration ended.
C
C   IT          - Number of iterations.
C
C   DS          - Step length (Usually length, but time if ISTEP = 2 or -2
C                 and DELTA not positive).
C
C   DS1         - Final step length to reach edge of cell.
C
C   DPATH       - Portion of step length clocked up
C
C   ISOLID      - =1,2,3 to identify if path is constrained to lie on
C                 P=0,1 or Q=0,1 or R=0,1. (ISOLIDS identifies if 0 or 1.)
C                 =0 if not constrained to surface.
C
C   ISOLIDS     - =0,1 if path constrained to P,Q or R 
C
C   TIMESTEPS   - Logical = .TRUE. if ISTEP = +2 or -2
C
*//* 26.03.93 New working variables
C
C   IBOCS(6)    - Stored values of IBOCC. Set to 0 when IBOCW set to 1 to
C                 activate tangency for relevant face.
C
C   IBOCW(6)    - Working vaues of IBOCC. Set to zero initially.
C
C   IBOCANY     - Logical = .TRUE. if any of IBOCW = 1
C
*//* End Mod

      DIMENSION GC(3,8),VELC(3,8)
      DIMENSION XPT(3,2),XLAST(3),PQR(3),UVW(3)

      LOGICAL TIMESTEPS

      COMMON/CELL/P,Q,R,DP,DQ,DR,DS1,DT,P1,Q1,R1,EX,NOUT,IREP

*//* 26.03.93 New working variables
      DIMENSION IBOCS(6),IBOCW(6)
      LOGICAL   IBOCANY
*//* End Mod

C  Copy GCI and VELCI to GC and VELC. If NDXC equals 2 duplicate data in 
C  K=1 plane.

      IF (NDX .NE. 3) STOP 'SLINC: NDX NOT EQUAL TO 3'

      DO I=1,NDX
         DO J=1,2**NDXC
            GC(I,J)  = GCI(I,J)
            VELC(I,J)= VELCI(I,J)
         ENDDO
      ENDDO
      ISOLID = 0                          ! Particle not constrained to face

      IF (NDXC .EQ. 2) THEN

         DO I=1,NDX
            DO J=1,4
               GC(I,J+4)=GCI(I,J)+0.1
               VELC(I,J+4)=VELCI(I,J)
            ENDDO
         ENDDO

         ISOLID = 3                       ! Particle constrained to K=0
         ISOLIDS = 0

      ELSEIF (NDXC .NE. 3) THEN

         STOP 'SLINC: INVALID NDXC'

      ENDIF       
 
      IF (ABS(ISTEP) .EQ. 2) THEN
         TIMESTEPS = .TRUE.
      ELSE
         TIMESTEPS = .FALSE.
      ENDIF 

*//* 26.03.93 Initialisation of IBOCS and IBOCW and IBOCANY included
*             and Label 100 introduced.

       DO I=1,2*NDXC
          IBOCS(I) = IBOCC(I)
          IBOCW(I) = 0
       ENDDO
       IBOCANY = .FALSE.    


100   IF (DELTA.GT.0) THEN

*//* End Mod

C       Set initial step length so that 10 steps would be needed to cross 
C       diagonally across cell.

         ICORN=8
 
         IF (ISOLID .GT. 0) ICORN=4
 
         DS=0
         DO J=1,3
            DS=DS+(GC(J,1)-GC(J,ICORN))**2
         ENDDO
         DS = 0.1 * SQRT(DS)

      ELSE

         DS=DL

      ENDIF

      IF (ISTEP .LT. 0) DS = -DS

      IT  = 0                   ! Number of iterations

C  Set P,Q,R to start point of streamline

1     P = XC1(1)
      Q = XC1(2)

      R = XC1(3)

      NP    = 0                 ! Number of path points
      DS1   = DS                ! FINAL STEP LENGTH
      DPATH = DL1               ! Path length to end of this step
      NOUT  = 0                 ! Cell face not reached
      NFLOP = 2                 ! Flip-flop index to working path points
      NCS   = 0                 ! Number of computational steps 
 
2     NFLOP = 3 - NFLOP         ! Change flip-flop
      NCS = NCS + 1             ! Number of steps in path      

      IF (NCS .GT. NISMAX) THEN
         IRTN = 2
         NP = 0
         RETURN
      ENDIF

      IREP = 0                  ! IREP=0 for first pass (using step start point)
                                ! IREP=1 for second pass (using step centre) 

C  Set P0,Q0,R0 to the start values of P,Q,R for this step. P,Q,R are then 
C  used as working variables for this step.

      P0 = P
      Q0 = Q
      R0 = R

C  If path constrained to lie on surface set P,Q,R accordingly


C  At moment only face R=0 can be solid

      IF (ISOLID .GT. 0) R=0

C  Get cartesian coordinates of point (P,Q,R)

      PQR(1)=P
      PQR(2)=Q
      PQR(3)=R
      CALL INTEVC (3,3,PQR,GC,XPT(1,NFLOP),IRTN)

C  If cell face reached end this path

      IF (NOUT.NE.0) GOTO 9


C  Unless predefined time intervals calculate time increment DT 

3     IF (DELTA.LE.0 .AND. ABS(ISTEP).EQ.2) THEN

         DT = DS
         GOTO 4

      ENDIF 

C   Find velocity at (P,Q,R)

      PQR(1)=P
      PQR(2)=Q
      PQR(3)=R

      CALL INTEVC (3,3,PQR,VELC,UVW,IRTN)

	U=UVW(1)
	V=UVW(2)
	W=UVW(3)

      VEL = SQRT(U**2+V**2+W**2)

      IF (VEL .EQ. 0) THEN
         IRTN = 5
         RETURN
      ENDIF

      DT = DS/VEL

4     CONTINUE
     

C CALCULATE THE CHANGES IN P,Q AND R DURING THE STEP.
C U.DT = XP.DP + XQ.DQ +XR.DR
C V.DT = YP.DP + YQ.DQ +YR.DR
C W.DT = ZP.DP + ZQ.DQ +ZR.DR
C
C WHERE XP( = dX/dP) ETC AND U,V AND W ARE CALCULATED
C AT P+DP/2,Q+DQ/2,R+DR/2
 
 
C If path constrained to lie on surface use INVERT2D

      IF (ISOLID .GT. 0) THEN

         CALL INVERT2D(GC(1,1),GC(1,2),GC(1,3),GC(1,4),
     &                 U, V, W,DT,DP,DQ,DR, P, Q, R, 0.0)

      ELSE

C Otherwise EQUATIONS ARE SOLVED BY DIRECT INVERSION OF THE 3x3 MATRIX
 
          B1 = U*DT
          B2 = V*DT
          B3 = W*DT

          CALL INVERT3D(GC(1,1),GC(1,2),GC(1,6),GC(1,5),
     1                  GC(1,3),GC(1,4),GC(1,8),GC(1,7),
     2                  P, Q, R,B1,B2,B3,DP,DQ,DR)
 
      ENDIF

      CONTINUE

C  Constrain path to become tangential to solid surfaces as specified by
C  parameters IBOCC

*//* 26.03.93 Modification of DP,DQ,DR made dependant on IBOCANY being true.
*             Also working version IBOCW used in place of IBOCC.
      IF (IBOCANY) THEN 
         IF (IBOCW(1) .EQ. 1) DP=P*DP
         IF (IBOCW(2) .EQ. 1) DP=(1-P)*DP
         IF (IBOCW(3) .EQ. 1) DQ=Q*DQ
         IF (IBOCW(4) .EQ. 1) DQ=(1-Q)*DQ
         IF (NDXC .GT. 2) THEN
            IF (IBOCW(5) .EQ. 1) DR=R*DR
            IF (IBOCW(6) .EQ. 1) DR=(1-R)*DR
         ENDIF
      ENDIF
*//* End Mod

      P = P0
      Q = Q0
      R = R0

      P1 = P + DP
      Q1 = Q + DQ
      R1 = R + DR

      IF (P1 .GT. 1.0) THEN
         EX = (1-P)/DP
         CALL CLIP
         P1 = 1.0
      ELSEIF (P1 .LT. 0.0) THEN
         EX = -P/DP
         CALL CLIP
         P1 = 0.0
      ENDIF

      IF (Q1 .GT. 1.0) THEN
         EX = (1-Q)/DQ
         CALL CLIP
         Q1 = 1.0
      ELSEIF (Q1 .LT. 0.0) THEN
         EX = -Q/DQ
         CALL CLIP
         Q1 = 0.0
      ENDIF

      IF (R1 .GT. 1.0) THEN
         EX = (1-R)/DR
         CALL CLIP
         R1 = 1.0
      ELSEIF (R1 .LT. 0.0) THEN
         EX = -R/DR
         CALL CLIP
         R1 = 0.0
      ENDIF

      IF (IREP .EQ. 0) THEN

         IREP = 1

         P = 0.5*(P0 + P1)
         Q = 0.5*(Q0 + Q1)
         R = 0.5*(R0 + R1)

         GOTO 3

      ENDIF


C  Check if output step length reached, and if so put point in array

      IF (TIMESTEPS) THEN
         DABS = ABS(DT)
      ELSE   
         DABS = ABS(DS1) 
      ENDIF

      DPATH = DPATH + DABS

5     CONTINUE

      IF (DPATH .GE. DL) THEN

         T0 = DPATH - DABS
         EX = (DL - T0) / DABS

         P = P0 + EX*DP
         Q = Q0 + EX*DQ
         R = R0 + EX*DR

         NP = NP + 1
         IF (NP .GT. NPMAX) THEN
            np = npmax
            IRTN = 3
            RETURN
         ENDIF

         XCP(1,NP) = P
         XCP(2,NP) = Q
         XCP(3,NP) = R

         CALL INTEVC (3,3,XCP(1,NP),GC,XP(1,NP),IRTN)

         DPATH = DPATH - DL
         GOTO 5
  
      ENDIF

      P=P1
      Q=Q1
      R=R1

      GOTO 2

C REACH CELL SIDE; TEST FOR CONVERGENCE:
C ON THE PREVIOUS ITERATION THROUGH THE CELL THE PATH
C MET THE CELL BOUNDARY AT XLAST,YLAST,ZLAST.
C IF THE NORMAL FROM THIS POINT TO THE CURRENT PATH,
C (WHOSE DIRECTION IS DEFINED BY THE LAST 2 POINTS ON IT)
C HAS LENGTH SNORM AND THE TOTAL PATH LENGTH WITHIN THIS CELL
C IS SPATH THEN CONVERGENCE IS ASSUMED IF SNORM/SPATH < DELTA
 
  9   IF(DELTA .LE. 0.0) GOTO 11   
      
      IT = IT+1
 
      IF (IT.GT.1) THEN
         ALEN2=0
         BLEN2=0
         CLEN =0 

         DO I=1,3
              
            ALEN2 = ALEN2 +(XPT(I,NFLOP)-XPT(I,3-NFLOP))**2
            BLEN2 = BLEN2 +(XPT(I,NFLOP)-XLAST(I))**2
            CLEN  = CLEN +(XPT(I,NFLOP)-XPT(I,3-NFLOP))*
     &                                    (XPT(I,NFLOP)-XLAST(I))
               
         ENDDO

         ALEN = SQRT(ALEN2)

         IF (ALEN.LT. 1.0E-30 .OR. ABS(CLEN).LT. 1.0E-30) THEN
            CLEN = 0.
         ELSE
            CLEN = CLEN/ALEN
         ENDIF

         SNORM2 = BLEN2 - CLEN*CLEN

         SPATH2 = ((NCS - 2)*DS+DS1)**2

         IF (SPATH2.LT.1E-30) THEN
            IRTN=1
            RETURN
         ENDIF 

         IF ((SNORM2/SPATH2).LT.DELTA**2) GOTO 11

      ENDIF
 
      DO I=1,3
         XLAST(I) = XPT(I,NFLOP)
      ENDDO
 
      DS = 0.5 * DS
      IF (IT.LT.100) GOTO 1

      IRTN=4

      RETURN

*//* 26.03.93 Test included for tangency to be enforced if exit occurs through 
*             solid face.  If so and condition exists, IBOCANY is set true, 
*             appropriate element of IBOCW set to 1 and code re-entered at
*             label 100.   

11    CONTINUE

      IF (P.EQ.0. .AND. IBOCS(1).EQ.1) THEN
         IBOCW(1) = 1
         IBOCS(1) = 0
  
      ELSEIF (P.EQ.1. .AND. IBOCS(2).EQ.1) THEN
         IBOCW(2) = 1
         IBOCS(2) = 0
  
      ELSEIF (Q.EQ.0. .AND. IBOCS(3).EQ.1) THEN
         IBOCW(3) = 1
         IBOCS(3) = 0
  
      ELSEIF (Q.EQ.1. .AND. IBOCS(4).EQ.1) THEN
         IBOCW(4) = 1
         IBOCS(4) = 0
  
      ELSEIF (R.EQ.0. .AND. IBOCS(5).EQ.1) THEN
         IBOCW(5) = 1
         IBOCS(5) = 0
  
      ELSEIF (R.EQ.1. .AND. IBOCS(6).EQ.1) THEN
         IBOCW(6) = 1
         IBOCS(6) = 0
      
      ELSE 
         XC2(1)=P
         XC2(2)=Q
         XC2(3)=R

         DL2=DPATH
      
         testdl=(ncs-2)*ds+ds1
         if (testdl.lt.1e-30) then
            irtn=1
            return
         endif 

         IRTN=0
         RETURN  

      ENDIF

      IBOCANY = .TRUE.
      GOTO 100
*//* End Mod

      END
 
C ----------------------------------------------------------------------

        
      SUBROUTINE CLIP

      COMMON/CELL/P,Q,R,DP,DQ,DR,DS1,DT,P1,Q1,R1,EX,NOUT,IREP

      DP = DP*EX
      DQ = DQ*EX
      DR = DR*EX
      P1 = P + DP
      Q1 = Q + DQ
      R1 = R + DR

      IF(IREP .EQ. 1) THEN      
         DS1 = DS1*EX
         DT = DT*EX
         NOUT=1
      ENDIF

      RETURN 

      END

C --------------------------------------------------------------------------
      SUBROUTINE INTEVC (NDX,NDXC,XC,VC,VX,IRTN)
C --------------------------------------------------------------------------

C  Given Cell Coordinates of point in XC, return Cartesian coordinates in VX 

C --------------------------------------------------------------------------

      DIMENSION XC(NDXC),VC(NDX,2**NDXC),VX(NDX)

      P = XC(1)

      IF (NDXC .EQ. 1) THEN
         DO I=1,NDX
            VX(I)=(1-P)*VC(I,1)+P*VC(I,2)
         ENDDO
         RETURN
      ENDIF
                   
      Q = XC(2)      

      IF (NDXC .EQ. 3) THEN
         R = XC(3)
      ELSE
         R=0
      ENDIF

      OMP = 1 - P
      OMQ = 1 - Q
      OMR = 1 - R

      A000 = OMP*OMQ*OMR
      A001 = OMP*OMQ* R
      A010 = OMP* Q *OMR 
      A011 = OMP* Q * R
      A100 =  P *OMQ*OMR
      A101 =  P *OMQ* R
      A110 =  P * Q *OMR 
      A111 =  P * Q * R

      DO I=1,NDX

         VX(I)=A000*VC(I,1)+A100*VC(I,2)+A010*VC(I,3)+A110*VC(I,4)

         IF (NDXC .EQ. 3)  VX(I) = VX(I) +
     &         A001*VC(I,5)+A101*VC(I,6)+A011*VC(I,7)+A111*VC(I,8)
      ENDDO

      RETURN

      END

C --------------------------------------------------------------------------


C ----------------------------------------------------------------------
      SUBROUTINE INVERT3D(XA,XB,XC,XD,XE,XF,XG,XH,
     2                    P, Q, R,B1,B2,B3,DP,DQ,DR)
C ----------------------------------------------------------------------
 
C  THIS SUBROUTINE CALCULATES THE CHANGES IN P,Q,R.
C  UNDER THE EQUATIONS:
C
C  B1 = XP.DP + XQ.DQ + XR.DR
C  B2 = YP.DP + YQ.DQ + YR.DR
C  B3 = ZP.DP + ZQ.DQ + ZR.DR
C
C  WHERE XP = dX/dP etc and
C
C  INVERT IS CALLED FROM CELL AND Bn REPRESENTS THE ORTHOGONAL
C  DISPLACEMENTS U.DT ETC
 
      DIMENSION XA(3),XB(3),XC(3),XD(3),XE(3),XF(3),XG(3),XH(3)
      DIMENSION XP(3),XQ(3),XR(3)

      OMP = 1 - P
      OMQ = 1 - Q
      OMR = 1 - R
 
      DO I=1,3

         XP(I) = OMQ*OMR*(XB(I)-XA(I)) + OMQ*R*(XC(I)-XD(I))+
     &             Q*OMR*(XF(I)-XE(I)) +   Q*R*(XG(I)-XH(I))
 
         XR(I) = OMQ*OMP*(XD(I)-XA(I)) + OMQ*P*(XC(I)-XB(I))+
     &             Q*OMP*(XH(I)-XE(I)) +   Q*P*(XG(I)-XF(I))
 
         XQ(I) = OMP*OMR*(XE(I)-XA(I)) + OMP*R*(XH(I)-XD(I))+
     &             P*OMR*(XF(I)-XB(I)) +   P*R*(XG(I)-XC(I))
 
      ENDDO

C EQUATIONS ARE SOLVED BY THE DIRECT INVERSION OF THE
C 3 x 3 MATRIX.
 
      A11 = XQ(2)*XR(3) - XR(2)*XQ(3)
      A12 = XR(1)*XQ(3) - XQ(1)*XR(3)
      A13 = XQ(1)*XR(2) - XR(1)*XQ(2)
      A21 = XR(2)*XP(3) - XP(2)*XR(3)
      A22 = XP(1)*XR(3) - XR(1)*XP(3)
      A23 = XR(1)*XP(2) - XP(1)*XR(2)
      A31 = XP(2)*XQ(3) - XQ(2)*XP(3)
      A32 = XQ(1)*XP(3) - XP(1)*XQ(3)
      A33 = XP(1)*XQ(2) - XQ(1)*XP(2)
 
      DETER = XP(1)*A11 + XQ(1)*A21 + XR(1)*A31
 
      DP = (A11*B1 + A12*B2 + A13*B3)/DETER
      DQ = (A21*B1 + A22*B2 + A23*B3)/DETER
      DR = (A31*B1 + A32*B2 + A33*B3)/DETER

      RETURN
      END
C ----------------------------------------------------------------------

C ----------------------------------------------------------------------
      SUBROUTINE INVERT2D(XA,XB,XE,XF,U, V, W,
     &                     DT,DP,DQ,DR,P,Q,R,RLIM)
C ----------------------------------------------------------------------
 
C
C THIS SUBROUTINE SELECTS THE CORRECT PAIR OF EQUATIONS TO BE INVERTED
C WHEN TRACING SURFACE STREAMLINES
C
C 'CORRECT' BEING DEFINED AS THE PAIR THAT YIELDS THE BIGGEST
C DETERMINANT
 
      DIMENSION XA(3),XB(3),XE(3),XF(3),XP(3),XQ(3)

      OMP = 1 - P
      OMQ = 1 - Q
 
      DR = 0
      R  = RLIM
 
      DO I=1,3
         XP(I) = OMQ*(XB(I)-XA(I)) + Q*(XF(I)-XE(I))
         XQ(I) = OMP*(XE(I)-XA(I)) + P*(XF(I)-XB(I))
      ENDDO

      DETXY = ABS(XP(1)*XQ(2) - XQ(1)*XP(2))
      DETXZ = ABS(XP(1)*XQ(3) - XP(3)*XQ(1))
      DETYZ = ABS(XP(2)*XQ(3) - XP(3)*XQ(2))
 
      BIGDET = AMAX1(DETXY,DETXZ,DETYZ)
 
      IF(BIGDET.EQ.DETXY) THEN
 
      CALL INV2DM(U,V,DT,XP(1),XQ(1),XP(2),XQ(2),DP,DQ)
 
      RETURN
      ENDIF
 
      IF (BIGDET.EQ.DETXZ) THEN
 
      CALL INV2DM(U,W,DT,XP(1),XQ(1),XP(3),XQ(3),DP,DQ)
 
      RETURN
      ENDIF
 
      IF (BIGDET.EQ.DETYZ) THEN
 
      CALL INV2DM(V,W,DT,XP(2),XQ(2),XP(3),XQ(3),DP,DQ)
      RETURN
      ENDIF
 
      END
 
C ----------------------------------------------------------------------

C ----------------------------------------------------------------------
      SUBROUTINE INV2DM(VEL1,VEL2,DT,A11,A12,A21,A22,DP,DQ)
C ----------------------------------------------------------------------
C
C THIS SUBROUTINE INVERTS A 2 x 2 MATRIX
C
      DTOJ = DT/(A11*A22 - A12*A21)
      DP   = DTOJ*(VEL1*A22 - VEL2*A12)
      DQ   = DTOJ*(VEL2*A11 - VEL1*A21)
C
      RETURN
C
      END
C ----------------------------------------------------------------------


