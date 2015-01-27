C===========================================================================
C     Linearinterpolation
C     ----------------------------------------------------------------------
C
      SUBROUTINE IPOL1P ( X, Y, X0, Y0, IMAX )
C
C                        --- Aufrufparameter ---
C     X     = Feld von x-Koordinaten x(i)
C     Y     = Feld von zugehoerigen Funktionswerten
C             ! Felder in aufsteigender Reihenfolge sortiert !
C     X0    = x-Koordinate des gesuchten y-Wertes
C --> Y0    = gesuchter z-Wert
C     IMAX  = Anzahl x-Werte
C
C                        --- Hilfsvariablen ---
C     IU    = Feldindex fuer Stuetzstelle x(IU) < x0
C     IO    = Feldindex fuer Stuetzstelle x(IO) > x0
C
      DOUBLE PRECISION X(500),Y(500),X0,Y0
      INTEGER  IMAX,IU,IO
C
C     ---------------------------------------------------------------------
C                                                         >>> P R I N T <<<
C      PRINT*,'Suche y(x) x = ',X0
C     ---------------------------------------------------------------------
C     Pruefen, ob X0 im Wertebereich liegt
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
C     Y(X0) interpolieren
C
20    Y0=(X0-X(IU))*(Y(IO)-Y(IU))/(X(IO)-X(IU))+Y(IU)
C     ---------------------------------------------------------------------
      RETURN
      END
