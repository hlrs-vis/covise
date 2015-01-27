      subroutine nxtr
     +	(RWK, IWK, iinit, reglabel, idone,IERR )
c these routine reads the (sub)grid coordinates

      INTEGER IERR
      CHARACTER*20 reglabel
      REAL RWK
      INTEGER IWK
      INTEGER iinit, idone
      
      DIMENSION RWK(70000000)
      DIMENSION IWK(5000000)

      LOGICAL INIT, DONE

      DONE = .FALSE.
      idone = 0

      INIT = .TRUE.
      IF ( iinit .EQ. 1 ) GOTO 60
      INIT = .FALSE.

      
C     Find Which Regions are Present in Memory
60    CALL TRNXTR (RWK, IWK, INIT, reglabel, DONE, IERR)
      IF (IERR .GT. 0) GOTO 999

      IF ( DONE .EQV. .FALSE. ) GOTO 999 
      idone = 1

999   CONTINUE

      END





