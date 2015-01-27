      subroutine scalar
     +	( label, field, dimg, part, RWK, IERR )
c these routine reads the scalar data

      PARAMETER (NR=100000)
      REAL RWK(NR)
      INTEGER IERR
      INTEGER part, dimg
      REAL field(dimg)
      CHARACTER *(*) label

C     Get the scalar field values
      CALL TRSCAL (label, field, dimg, part, RWK, IERR)
      IF (IERR .GT. 0) GOTO 999

999   CONTINUE

      END

