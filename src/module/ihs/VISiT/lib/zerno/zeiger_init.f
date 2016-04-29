C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE ZEIGER_INIT()
  
      implicit none     

      integer  nkd_mat

      include 'common.zer'
c     *****************************************************************

c     WICHTIG:  Die hier definierten Felder mœssen logisch zueinander
c               stimmen

c     *****************************************************************
      nkd_mat=2

      IF (ncd.eq.2) THEN

        nkd_fla=4
        nkd_kan=4
        eck(1,1)=1
        eck(1,2)=1
        eck(1,3)=nkd_mat

        eck(2,1)=1          
        eck(2,2)=nkd_mat 
        eck(2,3)=nkd_mat
      
        eck(3,1)=nkd_mat
        eck(3,2)=nkd_mat
        eck(3,3)=nkd_mat
      
        eck(4,1)=nkd_mat
        eck(4,2)=1          
        eck(4,3)=nkd_mat

c       Flaeche i=1
        fla_zeig(1,1)=1
        fla_zeig(1,2)=2

c       Flaeche i=nkd_mat
        fla_zeig(2,1)=4
        fla_zeig(2,2)=3

c       Flaeche j=1         
        fla_zeig(3,1)=1
        fla_zeig(3,2)=4

c       Flaeche j=nkd_mat
        fla_zeig(4,1)=2
        fla_zeig(4,2)=3

        fla_idie(1)=1
        fla_idie(2)=1
        fla_idie(3)=2
        fla_idie(4)=2

        fla_wert(1)=1
        fla_wert(2)=nkd_mat
        fla_wert(3)=1
        fla_wert(4)=nkd_mat


        kan_zeig(1,1)=1
        kan_zeig(1,2)=2

        kan_zeig(2,1)=4
        kan_zeig(2,2)=3

        kan_zeig(3,1)=1
        kan_zeig(3,2)=4

        kan_zeig(4,1)=2
        kan_zeig(4,2)=3


c       Die Laufvariable wird mit 0 gekennzeichnet:
        kan_idie(1,1)=1
        kan_idie(1,2)=0

        kan_idie(2,1)=nkd_mat
        kan_idie(2,2)=0

        kan_idie(3,1)=0         
        kan_idie(3,2)=1

        kan_idie(4,1)=0         
        kan_idie(4,2)=nkd_mat

      ENDIF
c     *****************************************************************

c     *****************************************************************
      IF (ncd.eq.3) THEN

        nkd_fla=6
        nkd_kan=12

        eck(1,1)=1
        eck(1,2)=1
        eck(1,3)=nkd_mat

        eck(2,1)=1          
        eck(2,2)=nkd_mat 
        eck(2,3)=nkd_mat
      
        eck(3,1)=nkd_mat
        eck(3,2)=nkd_mat
        eck(3,3)=nkd_mat
      
        eck(4,1)=nkd_mat
        eck(4,2)=1          
        eck(4,3)=nkd_mat

        eck(5,1)=1
        eck(5,2)=1
        eck(5,3)=1        

        eck(6,1)=1          
        eck(6,2)=nkd_mat 
        eck(6,3)=1        
      
        eck(7,1)=nkd_mat
        eck(7,2)=nkd_mat
        eck(7,3)=1        
      
        eck(8,1)=nkd_mat
        eck(8,2)=1          
        eck(8,3)=1        



c       Flaeche i=1
        fla_zeig(1,1)=1
        fla_zeig(1,2)=2
        fla_zeig(1,3)=6
        fla_zeig(1,4)=5

c       Flaeche i=nkd_mat
        fla_zeig(2,1)=4
        fla_zeig(2,2)=3
        fla_zeig(2,3)=7
        fla_zeig(2,4)=8

c       Flaeche j=1
        fla_zeig(3,1)=1
        fla_zeig(3,2)=5
        fla_zeig(3,3)=8
        fla_zeig(3,4)=4

c       Flaeche j=nkd_mat
        fla_zeig(4,1)=2
        fla_zeig(4,2)=6
        fla_zeig(4,3)=7
        fla_zeig(4,4)=3

c       Flaeche k=1
        fla_zeig(5,1)=5
        fla_zeig(5,2)=6
        fla_zeig(5,3)=7
        fla_zeig(5,4)=8

c       Flaeche k=nkd_mat
        fla_zeig(6,1)=1
        fla_zeig(6,2)=2
        fla_zeig(6,3)=3
        fla_zeig(6,4)=4

        fla_idie(1)=1
        fla_idie(2)=1
        fla_idie(3)=2
        fla_idie(4)=2
        fla_idie(5)=3
        fla_idie(6)=3

        fla_wert(1)=1
        fla_wert(2)=nkd_mat
        fla_wert(3)=1
        fla_wert(4)=nkd_mat
        fla_wert(5)=1
        fla_wert(6)=nkd_mat


        kan_zeig(1,1)=1
        kan_zeig(1,2)=2

        kan_zeig(2,1)=4
        kan_zeig(2,2)=3

        kan_zeig(3,1)=1
        kan_zeig(3,2)=4

        kan_zeig(4,1)=2
        kan_zeig(4,2)=3

        kan_zeig(5,1)=1
        kan_zeig(5,2)=5

        kan_zeig(6,1)=2
        kan_zeig(6,2)=6

        kan_zeig(7,1)=4
        kan_zeig(7,2)=8

        kan_zeig(8,1)=3
        kan_zeig(8,2)=7

        kan_zeig(9,1)=5
        kan_zeig(9,2)=6

        kan_zeig(10,1)=8
        kan_zeig(10,2)=7

        kan_zeig(11,1)=5
        kan_zeig(11,2)=8

        kan_zeig(12,1)=6
        kan_zeig(12,2)=7



c       Die Laufvariable der Kante wird mit 0 gekennzeichnet:
        kan_idie(1,1)=1
        kan_idie(1,2)=0
        kan_idie(1,3)=nkd_mat

        kan_idie(2,1)=nkd_mat
        kan_idie(2,2)=0
        kan_idie(2,3)=nkd_mat

        kan_idie(3,1)=0
        kan_idie(3,2)=1
        kan_idie(3,3)=nkd_mat

        kan_idie(4,1)=0
        kan_idie(4,2)=nkd_mat
        kan_idie(4,3)=nkd_mat

        kan_idie(5,1)=1
        kan_idie(5,2)=1
        kan_idie(5,3)=0       

        kan_idie(6,1)=1
        kan_idie(6,2)=nkd_mat
        kan_idie(6,3)=0       

        kan_idie(7,1)=nkd_mat
        kan_idie(7,2)=1
        kan_idie(7,3)=0       

        kan_idie(8,1)=nkd_mat
        kan_idie(8,2)=nkd_mat
        kan_idie(8,3)=0       

        kan_idie(9,1)=1
        kan_idie(9,2)=0
        kan_idie(9,3)=1

        kan_idie(10,1)=nkd_mat
        kan_idie(10,2)=0
        kan_idie(10,3)=1

        kan_idie(11,1)=0
        kan_idie(11,2)=1
        kan_idie(11,3)=1

        kan_idie(12,1)=0
        kan_idie(12,2)=nkd_mat
        kan_idie(12,3)=1
      ENDIF
c     *****************************************************************

      return
      end
