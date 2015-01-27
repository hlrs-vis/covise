      subroutine blocko
     +	( blck, coanz, part, RWK, IERR )
c these routine reads the (sub)grid coordinates

      PARAMETER (NR=100000)
      REAL RWK(NR)
      INTEGER IERR
      INTEGER part, coanz
      INTEGER blck(coanz)


C     Get dimensions of possible subgrid
      CALL TRBLOF (blck, coanz, part, RWK, IERR)
      IF (IERR .GT. 0) GOTO 999

999   CONTINUE

      END

