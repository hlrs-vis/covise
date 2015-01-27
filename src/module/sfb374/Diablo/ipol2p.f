C===========================================================================
C     Interpolation einer zweiparametrigen Funktion
C     ----------------------------------------------------------------------
C
      SUBROUTINE IPOL2P (X,Y,Z,X0,Y0,Z0,IMAX,JMAX)
C
C                        --- Aufrufparameter ---
C     X     = Feld von x-Koordinaten x(i)
C     Y     = Feld von y-Koordinaten y(i)
C     Z     = Feld von zugehoerigen Funktionswerten z(i,j) = z(x,y)
C             ! Felder in aufsteigender Reihenfolge sortiert !
C     X0    = x-Koordinate des gesuchten z-Wertes
C     Y0    = y-Koordinate des gesuchten z-Wertes
C --> Z0    = gesuchter z-Wert
C     IMAX  = Anzahl x-Werte
C     JMAX  = Anzahl y-Werte
C                        --- Hilfsvariablen ---
C     FX,FY = Hilfsvariablen fuer die Interpolation
C     IU    = Feldindex fuer Stuetzstelle x(IU) < x0
C     JU    = Feldindex fuer Stuetzstelle y(JU) < y0
C     IO    = Feldindex fuer Stuetzstelle x(IO) > x0
C     JO    = Feldindex fuer Stuetzstelle y(JO) > y0
C
      DOUBLE PRECISION X(100),Y(100),Z(100,100),X0,Y0,Z0,FX,FY
      INTEGER  IMAX,JMAX,IU,JU,IO,JO
C
C     ---------------------------------------------------------------------
C                                                         >>> P R I N T <<<
C      PRINT*,'Suche z(x,y) x,y=',X0,Y0
C     ---------------------------------------------------------------------
C     Pruefen, ob X0 im Wertebereich liegt, sonst Extrapolation
C
      IF (X0.LT.X(1)) THEN
	IU = 1
	IO = 2
	GOTO 20
      END IF
      IF (X0.GT.X(IMAX)) THEN
	IU = IMAX-1
	IO = IMAX
	GOTO 20
      END IF
C     ---------------------------------------------------------------------
C     IU und IO bestimmen
C
      IU = 0
      IO = 1
10    IU = IU+1
      IO = IO+1
      IF (X0.GT.X(IO)) THEN
	GOTO 10
      END IF
C     ---------------------------------------------------------------------
C     Pruefen, ob Y0 im Wertebereich liegt, sonst Extrapolation
C
20    IF (Y0.LT.Y(1)) THEN
	JU = 1
	JO = 2
	GOTO 40
      END IF
      IF (Y0.GT.Y(JMAX)) THEN
	JU = JMAX-1
	JO = JMAX
	GOTO 40
      END IF
C     ---------------------------------------------------------------------
C     JU und JO bestimmen
C
      JU = 0
      JO = 1
30    JU = JU+1
      JO = JO+1
      IF (Y0.GT.Y(JO)) THEN
	GOTO 30
      END IF
C     ---------------------------------------------------------------------
C     Z(X0,Y0) inter(-extra)polieren
C
40    FX = (X0-X(IU))/(X(IO)-X(IU))
      FY = (Y0-Y(JU))/(Y(JO)-Y(JU))
      Z0 =   (1-FX)*(1-FY)*Z(IU,JU) + FX*(1-FY)*Z(IO,JU)
     1         + (1-FX)*FY*Z(IU,JO) +     FX*FY*Z(IO,JO)
C     ---------------------------------------------------------------------
      RETURN
      END
C
C===========================================================================
