      subroutine rread
     +	(RWK, IWK, CWK, IERR)
c these routine reads the (sub)grid coordinates

      INTEGER IERR
      REAL RWK
      INTEGER IWK
      CHARACTER CWK

      DIMENSION RWK(70000000)
      DIMENSION IWK(5000000)
      DIMENSION CWK(50000)


C     Override the Default Filename and Format
      CALL TRREAD (RWK, IWK, CWK, IERR)
      IF (IERR .GT. 0) GOTO 999

999   CONTINUE

      END

