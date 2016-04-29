C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE COLOR_GEB(gemat,gemat_adr,nl_gemat,
     *                     perm_kn,perm_kn_inv,
     *                     folg,zeig,farb_kn,farb_anz,grad_kn,
     *                     farb_adr,nfarb_kn)

      implicit none

      include 'common.zer'

      integer gemat,gemat_adr,nl_gemat,
     *        perm_kn,perm_kn_inv

      integer folg,zeig,farb_kn,farb_anz,grad_kn,farb_adr

      integer i,k,j,iii,i_wieder,grad,grad_max,grad_min,
     *        nfolg,ngrenz_cor,
     *        ikn,ipoin,nfarb_alt,n_wieder,cor,ifarb,
     *        anz_min,anz_max,start_node,neuer_node,luerr,
     *        farb_min,farb_max,kflag,nkn,nnn,
     *        nfarb_kn

      integer icom,lentb,igeb,anz

      character*80 comment,zeil_1,zeil_2

      parameter (n_wieder=5)
      
      dimension gemat(nl_gemat),gemat_adr(ngebiet+1),
     *          perm_kn(ngebiet),perm_kn_inv(ngebiet),
     *          zeig(ngebiet),farb_kn(ngebiet),folg(ngebiet),
     *          farb_anz(ngebiet),grad_kn(ngebiet),
     *          farb_adr(ngebiet+1)
c     *****************************************************************

c     do 12 i=1,ngebiet
c       write(lupar,13) i,(gemat(j),j=gemat_adr(i),gemat_adr(i+1)-1)
c       write(6,13) i,(gemat(j),j=gemat_adr(i),gemat_adr(i+1)-1)
c12   continue
c13   format(i3,4x,10(i3,1x))


c     farb_kn.........Farbe jedes Knoten                             
c     zeig............Zeiger ob Knoten schon coloriert ist oder nicht
c     fab_anz.........Anzahl Knoten pro Farbe                         

      do 10 i=1,ngebiet
        farb_kn(i)=0
        zeig(i)=0
        folg(i)=0
        perm_kn_inv(i)=0
        perm_kn(i)=0
        farb_anz(i)=0
 10   continue


c     *****************************************************************
c     KNOTEN-COLORIERUNG:                                 
 
c     Als Start-Knoten wird der Knoten mit den meisten Nachbarn
c     verwendet:

      grad_max=0
      grad_min=10000000
      do 100 i=1,ngebiet
         grad=gemat_adr(i+1)-gemat_adr(i)
         if (grad.gt.grad_max) then
           start_node=i
           grad_max=grad
         endif
         grad_min=MIN(grad_min,grad)
 100  continue


c     Zu Beginn wird die Farben-Anzahl der Anzahl maximaler 
c     Nachbarn gesetzt:
      nfarb_kn=grad_max 

      do 1000 i_wieder=1,n_wieder            

c         Wiederholung der Colorierung ist notwendig wenn waehrend
c         der Colorierung neue Farben eingefuehrt werden muessen.
c         Dann koennen die Knotenanzahlen pro Farbe stark 
c         unterschiedlich sein. -> Wenn die bereits bestimmte
c         mindest notwendige Farbenanzahl nfarb_alt mit der
c         Farbenanzahl nfarb_kn am Ende der Colorierung uebereinstimmt
c         ist alles in Ordnung. Ansonsten wird die Colorierung
c         mit erhoehter Farbenanzahl wiederholt

          nfarb_alt=nfarb_kn

          do 202 i=1,ngebiet
            farb_kn(i)=0
            zeig(i)=0
            farb_anz(i)=0
            folg(i)=0
 202      continue


c         Colorierung des Start-Knotens und seiner Nachbarn:
          nfolg=1
          cor=1
          folg(nfolg)=start_node
          farb_kn(start_node)=cor
          farb_anz(cor)=farb_anz(cor)+1
          do 203 j=gemat_adr(start_node),gemat_adr(start_node+1)-1
             cor=cor+1
             nfolg=nfolg+1      
             folg(nfolg)=gemat(j)
             farb_kn(gemat(j))=cor    
             farb_anz(cor)=farb_anz(cor)+1
 203      continue
          nfarb_kn=cor

          if (nfolg.eq.ngebiet) then
c             Alle Knoten sind coloriert

c             Kontrolle ob neue Farben eingefuehrt wurden
              if (nfarb_kn.eq.nfarb_alt) then
c                 Colorierung ist abgeschlossen
                  goto 1001
              else 
c                 Es wurden neue Farben eingefuehrt. Daher
c                 muss Colorierung wiederholt werden.
                  goto 999
              endif
          endif 
          
          do 200 iii=1,ngebiet

               ikn=folg(iii)


               IF (ikn.eq.0) THEN
c                 Graph ist nicht zusammenhaengend -> Startknotensuche
                  grad_max=0
                  grad_min=10000000
                  do 112 i=1,ngebiet
                     if (farb_kn(i).eq.0) then
                        grad=gemat_adr(i+1)-gemat_adr(i)
                        if (grad.gt.grad_max) then
                          neuer_node=i
                          grad_max=grad
                        endif
                        grad_min=MIN(grad_min,grad)
                     endif
 112              continue
                  nfolg=nfolg+1
                  nfarb_kn=nfarb_kn+1
                  cor=nfarb_kn
                  folg(nfolg)=neuer_node
                  farb_kn(neuer_node)=cor
                  farb_anz(cor)=farb_anz(cor)+1
                  ikn=neuer_node

                  if (nfolg.eq.ngebiet) then
c                        Alle Knoten sind coloriert

c                        Kontrolle ob neue Farben eingefuehrt wurden
                         if (nfarb_kn.eq.nfarb_alt) then
c                            Colorierung ist abgeschlossen
                             goto 1001
                         else 
c                            Es wurden neue Farben eingefuehrt. Daher
c                            muss Colorierung wiederholt werden.
                             goto 999
                         endif
                  endif 
               ENDIF

c              Colorierung aller am Knoten ikn angrenzender Knoten:
               do 210 j=gemat_adr(ikn),gemat_adr(ikn+1)-1

                  ipoin=gemat(j)
                  IF (farb_kn(ipoin).eq.0) THEN

c                    Knoten ipoin besitzt noch keine Farbe

                     do 215 k=1,nfarb_kn
                       zeig(k)=0
 215                 continue


c                    Bestimmung der an Knoten ipoin angrenzenden Farben:
                     ngrenz_cor=0
                     do 220 k=gemat_adr(ipoin),gemat_adr(ipoin+1)-1
                          cor=farb_kn(gemat(k))

                          if (cor.ne.0) then
                             if (zeig(cor).eq.0) then
c                               Farbe cor wurde noch nicht als 
c                               Grenzfarbe geschrieben 
c                               
                                 ngrenz_cor=ngrenz_cor+1
                                 zeig(cor)=1
                             endif                         
                          endif
 220                 continue

                     if (ngrenz_cor.eq.0) then
                        call erro_init(myid,parallel,luerr)
                        write(luerr,*)'Fehler in Routine COLOR_GEB'
                        write(luerr,*)'Kein am Gebiet ',ipoin
                        write(luerr,*)'beteiligtes Gebiet ist '
                        write(luerr,*)'coloriert.             '
                        call erro_ende(myid,parallel,luerr)
                     endif

c                    Einfuehren einer neuen Farbe sofern die bisherige
c                    Farbenanzahl der Anzahl an Grenzfarben entspricht
                     if (ngrenz_cor.eq.nfarb_kn) then
                       nfarb_kn=nfarb_kn+1
                     endif

c                    Auswahl einer Farbe fuer Knoten ipoin. Dabei wird
c                    von den moeglichen Farben die ausgesucht die
c                    am wenigsten Knoten besitzt:

                     anz_min=100000000
                     do 250 ifarb=1,nfarb_kn

                        if (zeig(ifarb).eq.0) then
c                          Die Farbe ifarb ist eine moeglich Farbe, da
c                          keiner der an Knoten ipoin angrenzenden
c                          Knoten diese Farbe besitzt.
c                        
                           if (farb_anz(ifarb).lt.anz_min) then
                             cor=ifarb
                             anz_min=farb_anz(ifarb)
                           endif

                        endif
 250                 continue
c                    
                     nfolg=nfolg+1
                     folg(nfolg)=ipoin        
                     farb_kn(ipoin)=cor
                     farb_anz(cor)=farb_anz(cor)+1
  
                     if (nfolg.eq.ngebiet) then
c                        Alle Knoten sind coloriert

c                        Kontrolle ob neue Farben eingefuehrt wurden
                         if (nfarb_kn.eq.nfarb_alt) then
c                            Colorierung ist abgeschlossen
                             goto 1001
                         else 
c                            Es wurden neue Farben eingefuehrt. Daher
c                            muss Colorierung wiederholt werden.
                             goto 999
                         endif
                     endif 


                  ENDIF

 210           continue
 200      continue


 999  continue
1000  continue                               

      call erro_init(myid,parallel,luerr)
      write(luerr,*)'Fehler in Routine COLOR_GEB'
      write(luerr,*)'Nach ',n_wieder,' Wiederholung des '
      write(luerr,*)'Colorierungs-Algorithmus konnte '
      write(luerr,*)'keine Colorierung gefunden werden '
      call erro_ende(myid,parallel,luerr)

c     Colorierung ist abgeschlossen:
1001  continue
c     *****************************************************************

c     write(6,*)'Farbdaten:'
c     do 111 i=1,ngebiet
c        write(6,13) folg(i),farb_kn(i),farb_anz(i)
c111  continue


c     *****************************************************************
c     SORTIERNEN DER KNOTEN NACH FARBEN UND INNERHALB DER FARBE NACH
c     DEM GRAD BZW. NACH ANZAHL MATRIX-EINTRAEGEN PRO ZEILE:

      anz_min=10000000
      anz_max=-10000000

      nkn=0
      farb_adr(1)=1
      do 400 ifarb=1,nfarb_kn

         nnn=0
         do 410 i=1,ngebiet
             if (farb_kn(i).eq.ifarb) then
                nnn=nnn+1
                folg(nnn)=i
                zeig(nnn)=gemat_adr(i+1)-gemat_adr(i)
             endif
 410     continue

c        Sortieren der Knoten nach abnehmendem Grad der auf Feld zeig
c        gespeichert ist:
         kflag=-2
         CALL isort(zeig,folg,nnn,kflag)

        do 420 i=1,nnn
           nkn=nkn+1
           perm_kn_inv(nkn)=folg(i)
           grad_kn(nkn)=zeig(i)
 420    continue
        farb_adr(ifarb+1)=farb_adr(ifarb)+nnn  

         if (nnn.ne.farb_anz(ifarb)) then
          call erro_init(myid,parallel,luerr)
          write(luerr,*)'Fehler in Routine COLOR_GEB'
          write(luerr,*)'Die Anzahl markierter Gebiete von Farbe ',ifarb
          write(luerr,*)'stimmt mit der Anzahl gezaehlter Gebiete'
          write(luerr,*)'von Farbe ',ifarb,' nicht ueberein.'
          write(luerr,*)'Anzahl markierter Gebiete:',nnn   
          write(luerr,*)'Anzahl gezaehlter Gebiete:',farb_anz(ifarb)
          call erro_ende(myid,parallel,luerr)
         endif


c        Bestimmen der maximalen und minimalen Knotenanzahlen pro Farbe
         if (farb_anz(ifarb).lt.anz_min) then
            anz_min=farb_anz(ifarb)
            farb_min=ifarb
         endif
         if (farb_anz(ifarb).gt.anz_max) then
            anz_max=farb_anz(ifarb)
            farb_max=ifarb
         endif

 400  continue

      if (nkn.ne.ngebiet) then
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Fehler in Routine COLOR_GEB'
        write(luerr,*)'Die Summe aller Gebiete pro Farbe stimmt  '
        write(luerr,*)'mit der Gebietsanzahl nicht ueberein '
        write(luerr,*)'Summe aller Gebiete pro Farbe:',nkn 
        write(luerr,*)'Gebietsanzahl               :',ngebiet
        call erro_ende(myid,parallel,luerr)
      endif


c     Belegen des Spalten-Permutationsvektors:
      do 430 i=1,ngebiet
         perm_kn(perm_kn_inv(i))=i
 430  continue
c     *****************************************************************


c     *****************************************************************
c     AUSDRUCK DER GEBIETSGRAPHEN:

      do 120 i=2,80 
         zeil_1(i-1:i)='*'
         zeil_2(i-1:i)='-'
 120  continue
 777  format(1x,A70)
 666  format(1x,A)
 555  format(1x,i3,3x,i2,6x,i2,4x,100(i3,1x))

      comment='Gebietsgraph der ausgewaehlten Zerlegung'
      icom=lentb(comment)
      write(lupro,*)                   
      write(lupro,777) zeil_1       
      write(lupro,666) comment(1:icom)
      write(lupro,666) zeil_2(1:icom)       

      write(lupro,*)'Nr.  Anzahl  Farbe             Graph  '
      do 600 igeb=1,ngebiet
         anz=gemat_adr(igeb+1)-gemat_adr(igeb)
         write(lupro,555) igeb,anz,farb_kn(igeb),(gemat(k),
     *                   k=gemat_adr(igeb),gemat_adr(igeb+1)-1)
 600  continue

      write(lupro,777) zeil_1       
c     *****************************************************************


c     *****************************************************************
c     AUSDRUCK:
c
c     write(lupar,*)'geb_num  farbe  grad   graph '
c     do 14 ifarb=1,nfarb_kn
c        do 13 iii=farb_adr(ifarb),farb_adr(ifarb+1)-1
c          igln_alt=perm_kn_inv(iii)
c          write(lupar,111) igln_alt,farb_kn(igln_alt),
c    *                      grad_kn(iii),
c    *        (gemat(j),j=gemat_adr(igln_alt),gemat_adr(igln_alt+1)-1)
c13      continue
c14   continue
c111  format(3(i3,1x),3x,30(i3,1x))
c     *****************************************************************

      return
      end
