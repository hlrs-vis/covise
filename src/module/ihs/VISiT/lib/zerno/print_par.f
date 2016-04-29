C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE PRINT_PAR(comment,geom_dim,rand_dim,ndat_max)

      implicit none

      include 'common.zer'

      integer   rand_dim,geom_dim,ndat_max,icom,j,i,luerr,lentb

      character*80 comment
      character*42 zeil_geo(12),zeil_rbe(10),zeil_kom(10)
      character*120 text_1,text_2

      dimension rand_dim(ngebiet,ndat_max),geom_dim(ngebiet,ndat_max)
c     ****************************************************************

      
c     ****************************************************************
      zeil_geo(1)='Prozessor'
      zeil_geo(2)='npoin'
      zeil_geo(3)='ncore'
      zeil_geo(4)='nhalo'
      zeil_geo(5)='nelem'
      zeil_geo(6)='ncore'
      zeil_geo(7)='nhalo'

      zeil_rbe(1)='Prozessor'
      zeil_rbe(2)='ndisp'
      zeil_rbe(3)='nwand'
      zeil_rbe(4)='npres'
      zeil_rbe(5)='nsyme'
      zeil_rbe(6)='nzykl'
      zeil_rbe(7)='nconv'
      zeil_rbe(8)='ntemp'                   

        do 120 i=2,120
         text_1(i-1:i)='*'
         text_2(i-1:i)='-'
 120    continue

      if (myid.eq.0) then

          icom=lentb(comment)
          write(lupro,*)
          write(lupro,777) text_1
          write(lupro,555)comment(1:icom)
          write(lupro,555)text_2(1:icom)

          if (parti_geo) then
             write(lupro,*) 
             write(lupro,888) (zeil_geo(j),j=1,7)
             do 201 i=1,ngebiet 
               write(lupro,999) i,(geom_dim(i,j),j=1,6)
 201         continue
          endif

          if (parti_rbe) then
             write(lupro,*) 
             write(lupro,888) (zeil_rbe(j),j=1,8)
             do 301 i=1,ngebiet
               write(lupro,999) i,(rand_dim(i,j),j=1,7)
 301         continue
          endif

          write(lupro,777) text_2

          zeil_geo(1)='Maximale Knotennummer :'
          zeil_geo(2)='Gesamtanzahl Knoten   :'
          zeil_rbe(1)='Maximale Elementnummer:'
          zeil_rbe(2)='Gesamtanzahl Elemente :'
          zeil_kom(1)='Anzahl doppelte Freiheitsgrade:'
          zeil_kom(2)='Anzahl doppelte Elemente      :'

          write(lupro,666) zeil_geo(1),knmax_num,zeil_geo(2),npoin_ges
          write(lupro,666) zeil_rbe(1),elmax_num,zeil_rbe(2),nelem_ges

          write(lupro,777) text_1
          write(lupro,*)

 444      format(1x,2(A31,1x,i8,2x))
 666      format(1x,2(A23,1x,i8,2x))
 777      format(1x,A70)
 555      format(1x,A)
 888      format(1x,A9,4x,10(A5,3x))

 999      format(1x,i3,8x,10(i7,1x))
 998      format(1x,i3,8x,3(i7,1x),8x,3(i7,1x))

 885      format(1x,A9,4x,2(A5,3x),2(A6,2x))

      endif
c     ****************************************************************

      return
      end
