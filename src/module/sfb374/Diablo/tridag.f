C                                                                      |
      SUBROUTINE TRIDAG (a,b,c,r,u,N)
C
C    Loest eine Tridiagonal-Matrix
C    _                                   _     _     _       _     _
C   |  b1  c1  0   ...                    |   |  u1   |     |  r1   |
C   |  a2  b2  c2  ...                    |   |  u2   |     |  r2   |
C   |  0   a3  b3  ...                    |   |  u3   |     |  r3   |
C   |              ...                    | ž |  ...  |  =  |  ...  |
C   |              ...  bN-2  cN-2  0     |   |  uN-2 |     |  rN-2 |
C   |              ...  aN-1  bN-1  cN-1  |   |  uN-1 |     |  rN-1 |
C   |_                  0     aN    bN   _|   |_ uN  _|     |_ rN  _|
C
      integer N, NMAX, J
      PARAMETER (NMAX=81)
      double precision GAM(NMAX), a(N), b(N), c(N), r(N), u(N), BET
C
C  a    = Untere Diagonale (array)
C  b    = Hauptdiagonale (array)
C  a    = Obere Diagonale (array)
C  r    = Rechte Seite (array)
C  u    = Loesungsvektor (array)
C  N    = Anzahl der Gleichungen
C  NMAX = max. Anzahl der Gleichungen (vorgegeben durch diablo)
C
      IF (b(1).EQ.0.) PAUSE 'tridag : rewrite equations'
      BET = b(1)
      u(1) = r(1)/BET
      DO 11 J = 2,N
	GAM(J) = c(J-1)/BET
	BET = b(J)-a(J)*GAM(J)
	IF (BET.EQ.0.) PAUSE  'tridag failed'
	u(J) = (r(J)-a(J)*u(J-1))/BET
11    CONTINUE
      DO 12 J = N-1,1,-1
	u(J) = u(J)-GAM(J+1)*u(J+1)
12    CONTINUE
      RETURN
      END
