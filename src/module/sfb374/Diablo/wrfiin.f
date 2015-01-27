C=======================================================================
C     FIDAP-Eingabedatei "FDREAD.diff" fr aktuellen Knoten erstellen
C     ------------------------------------------------------------------
      SUBROUTINE WRFIIN (LKORN, FMIKRO, FMESH, NSTEPS, TAU1, TAU2,
     1                   TEMP1, ERRFLG)
C     ------------------------------------------------------------------
C   * AUFRUF-Parameter:
C i   LKORN    = Korngr”áe
C i   FMIKRO   = Mikrostrukturfaktor
C i   FMESH    = Anzahl Netzunterteilungen
C i   NSTEPS   = Anzahl zu rechnender Zeitschritte
C i   TAU1     = Startzeit
C i   TAU2     = Ende der Diffusionsrechnung
C i   TEMP1    = Anfangstemperatur (=Temp. im Schnittpkt. mit ZTA-Diag.)
C o   ERRFLG   = Fehler-Code: 1, wenn Fehler ; sonst unver„ndert
C               
      DOUBLE PRECISION LKORN, FMIKRO, TAU1, TAU2, TEMP1
      INTEGER          FMESH, NSTEPS, ERRFLG
C
C     ------------------------------------------------------------------
C   * Interne Variablen:
C     LMIKRO   = Gr”áe der Mikrostruktur = lKorn*fMikro
C     DTAU     = L„nge eines Zeitschrittes = (TAU2-TAU1)/NSTEPS
C     STRG     = Hilfs-String zum Lesen/Schreiben
C
      DOUBLE PRECISION LMIKRO, DTAU
      CHARACTER        STRG*100
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
      PRINT*,'FIDAP-Eingabedatei wird erstellt'
C
C     ------------------------------------------------------------------
C     Dateien ”ffnen
C
      OPEN (10,FILE='FDREAD.diff.std')
      OPEN (11,FILE='FDREAD.diff')
C
C     ------------------------------------------------------------------
C     "FDREAD.diff.std" nach "FDREAD.diff" kopieren
C     Anstelle der Zeile "#" die aktuellen Werte eintragen
C
10    READ (10,'(a1)',ERR=200,END=200) STRG
      IF (STRG.EQ.'#') THEN
	GOTO 20
      ENDIF
      BACKSPACE 10
      READ (10,'(a75)',ERR=200,END=200) STRG
      WRITE (11,'(a75)',ERR=210) STRG
      GOTO 10
20    LMIKRO = FMIKRO*LKORN
      WRITE (11,'(a10,e15.9)',ERR=210) '$lMikro = ', LMIKRO
      WRITE (11,'(a10,i3)',ERR=210)    '$fMesh  = ', FMESH
      WRITE (11,'(a10,e15.9)',ERR=210) '$tstart = (e15.9)', TAU1
      WRITE (11,'(a10,i3)',ERR=210) '$nsteps = (i3)', NSTEPS
      DTAU = (TAU2-TAU1)/NSTEPS
      WRITE (11,'(a10,e15.9)',ERR=210) '$dt     = ', DTAU
      WRITE (11,'(a10,e15.9)',ERR=210) '$TAc3   = ', TEMP1
30    READ (10,'(a75)',ERR=200,END=300) STRG
      WRITE (11,'(a75)',ERR=210) STRG
      GOTO 30
C
C       ----------------------------------------------------------------
C       Lesefehler in "FDREAD.diff.std"
C                                                      >>> E R R O R <<<
200     PRINT*
	PRINT*,'  Fehler beim Lesen der Standard-Eingabedatei !!!'
	CLOSE(10,STATUS='KEEP')
	CLOSE(11,STATUS='KEEP')
	ERRFLG = 1
	RETURN
C
C       ----------------------------------------------------------------
C       Schreibfehler in "FDREAD.diff"
C                                                      >>> E R R O R <<<
210     PRINT*
	PRINT*,'  Fehler beim Schreiben der Eingabedatei !!!'
	CLOSE(10,STATUS='KEEP')
	CLOSE(11,STATUS='KEEP')
	ERRFLG = 1
	RETURN
C
C     ------------------------------------------------------------------
C                                                      >>> P R I N T <<<
300   CLOSE(10,STATUS='KEEP')
      CLOSE(11,STATUS='KEEP')
      PRINT*,'beendet.'
      RETURN
      END
C
C=======================================================================
