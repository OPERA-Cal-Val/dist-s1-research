#! /usr/bin/env python

import urllib.request
import sys
import math
import datetime

def getDISTALERTStatus_vI(mapsource,ANNname,ids,skipNodata=False):
    #print(skipNodata)
    mapalert = {}
    for ID in ids:
        mapalert[ID] = [255 for i in range(0,366)]
        print(f"ID = {ID}")
        #with open(mapsource+'/'+ID+'_DIST-ALERT_'+ANNname+'.csv','r') as mapfile:
        with urllib.request.urlopen(mapsource+'/'+ID+'_DIST-ALERT_'+ANNname+'.csv') as mapfile:
            lines = mapfile.readlines()
            header = lines[0]
            maplist = lines[1:]
            for line in maplist:
                #(granuleID,SensingTime,ProductionTime,VEGDISTSTATUS,VEGANOM,VEGIND,VEGHIST,VEGANOMMAX,VEGDISTCONF,VEGDISTDATE,VEGDISTCOUNT,VEGDISTDUR,VEGLASTDATE,GENDISTSTATUS,GENANOM,GENANOMMAX,GENDISTCONF,GENDISTDATE,GENDISTCOUNT,GENDISTDUR,GENLASTDATE)= line.strip().split(',')
                (granuleID,SensingTime,ProductionTime,VEGDISTSTATUS,VEGANOM,VEGIND,VEGHIST,VEGANOMMAX,VEGDISTCONF,VEGDISTDATE,VEGDISTCOUNT,VEGDISTDUR,VEGLASTDATE,GENDISTSTATUS,GENANOM,GENANOMMAX,GENDISTCONF,GENDISTDATE,GENDISTCOUNT,GENDISTDUR,GENLASTDATE)= line.strip().decode('utf-8').split(',')
                day = dayDiff("20211001",SensingTime)
                
                if not (skipNodata and int(VEGIND)==255):
                    if int(VEGDISTSTATUS) in [1,2,3,7] and int(VEGANOMMAX) < 10:
                        mapalert[ID][day] = 0
                    else:
                        mapalert[ID][day] = int(VEGDISTSTATUS)

    return mapalert

def dayDiff(start,end):
  startdate = datetime.datetime.strptime(start,"%Y%m%d")
  enddate = datetime.datetime.strptime(end,"%Y%m%d")
  days = enddate-startdate
  return (days.days+1)

def getRefALERTbinaryDaily(yeslabels,nolabels):
  refalert = {}
  with open('referenceTimeSeries_last.csv','r') as mapfile:
    lines = mapfile.readlines()
    header = lines[0]
    reflist = lines[1:]
  for line in reflist:
    fields = line.strip().split(',')
    (ID,overallLabel,Long,Lat,changetype) = fields[0:5]
    daily = fields[5:]
    refalert[ID] = [0 for i in range(0,365)]
    for day in range(0,365):
      found = False
      for l in yeslabels:
        if l == daily[day]:
          found = True
        #if l in daily[day] and int(ID) in refVLmaj:
        #  found = True
      if found:
        refalert[ID][day] = 2
      else:
        for l in nolabels:
          if l == daily[day]:
            found = True
        if found:
          refalert[ID][day] = 1
        else:
          refalert[ID][day] = 0
  return refalert

def filterMap(ids,map,cat):
    mapout = {}
    #if cat == "lt50":
    #    nodist = [0]
    #    dist = [1,2,3,4,5,6]
    #    old = [7,8]
    #if cat == "gt50":
    #    nodist = [0,1,2,3,7]
    #    dist = [4,5,6]
    #    old = [8]
    #if cat == "confgt50":
    #    nodist = [0,1,2,3,7]
    #    dist = [6]
    #    old = [8]
    #if cat == "conflt50":
    #    nodist = [0]
    #    dist = [3,6]
    #    old = [7,8]
    if cat == "lt50":
        nodist = [0]
        dist = [1,2,3,4,5,6]
        old = [7,8]
    elif cat == "lt50_curr":
        nodist = [0]
        dist = [1,2,3,4,5,6]
        old = []
    elif cat == "provlt50":
        nodist = [0]
        dist = [2,3,5,6]
        old = [7,8]
    elif cat == "lt50_onlyprov":
        nodist = [0]
        dist = [2,5]
        old = []
    elif cat == "onlylt50":
        nodist = [0]
        dist = [1,2,3]
        old = [7]
    elif cat == "gt50":
        nodist = [0,1,2,3,7]
        dist = [4,5,6]
        old = [8]
    elif cat == "gt50_curr":
        nodist = [0,1,2,3,7]
        dist = [4,5,6]
        old = []
    elif cat == "provgt50":
        nodist = [0,1,2,3,7]
        dist = [5,6]
        old = [8]
    elif cat == "gt50_onlyprov":
        nodist = [0,1,2,3,7]
        dist = [5]
        old = []
    elif cat == "confgt50":
        nodist = [0,1,2,3,7]
        dist = [6]
        old = [8]
    elif cat == "provgt50_curr":
        nodist = [0,1,2,3,7]
        dist = [5,6]
        old = []  
    elif cat == "confgt50_curr":
        nodist = [0,1,2,3,7]
        dist = [6]
        old = []
    elif cat == "conflt50":
        nodist = [0]
        dist = [3,6]
        old = [7,8]
    elif cat == "conflt50_curr":
        nodist = [0]
        dist = [3,6]
        old = []
    else:
        print(cat, "not found")
    for ID in ids:
        mapout[ID] = [0 for x in range(0,366)]
        for d in range(0,366):
            if map[ID][d] in nodist:
                mapout[ID][d] = 1
            elif map[ID][d] in dist:   
                mapout[ID][d] = 2
            elif map[ID][d] in old:
                mapout[ID][d] = 3
            else:
                mapout[ID][d] = 0
    return mapout

def alertConfusionMatrix_vTS2(ids,cat,map,ref,strata,strataCounts,mincount,excludelist,Ndays,name,printMatrix = False):
  N = strataCounts
  Nstrata = len(strataCounts)
  Ntotal = sum([N[str(s)] for s in range(1,Nstrata)])
  n = [[[0,0,0],[0,0,0],[0,0,0]] for s in range(Nstrata+1)]
  ntotal = [0 for s in range(Nstrata+1)]
  with open(name+".txt",'w') as OUT:
        "ID,agree_no,commission,omission,agree_loss,total\n"
  #enum Status {NODIST=0,FIRSTLO=1, PROVLO=2,CONFLO=3,FIRSTHI=4,PROVHI=5,CONFHI=6,CONFLOFIN=7,CONFHIFIN=8,NODATA=255};

  if cat == "lt50":
        nodist = [0]
        dist = [1,2,3,4,5,6]
        old = [7,8]
  if cat == "provlt50":
        nodist = [0]
        dist = [2,3,5,6]
        old = [7,8]
  if cat == "lt50_onlyprov":
        nodist = [0]
        dist = [2,5]
        old = []
  if cat == "onlylt50":
        nodist = [0]
        dist = [1,2,3]
        old = [7]
  if cat == "gt50":
        nodist = [0,1,2,3,7]
        dist = [4,5,6]
        old = [8]
  if cat == "provgt50":
        nodist = [0,1,2,3,7]
        dist = [5,6]
        old = [8]
  if cat == "gt50_onlyprov":
        nodist = [0,1,2,3,7]
        dist = [5]
        old = []
  if cat == "confgt50":
        nodist = [0,1,2,3,7]
        dist = [6]
        old = [8]
  if cat == "provgt50_curr":
        nodist = [0,1,2,3,7]
        dist = [5,6]
        old = []  
  if cat == "confgt50_curr":
        nodist = [0,1,2,3,7]
        dist = [6]
        old = []
  if cat == "conflt50":
        nodist = [0]
        dist = [3,6]
        old = [7,8]
  if cat == "conflt50_curr":
        nodist = [0]
        dist = [3,6]
        old = []
  #confusion matrix
  for ID in ids:
    p = [[0,0,0],[0,0,0],[0,0,0]]
    ptotal = 0
    for d in range(0,365):
      #print(ref[ID][d], map[ID][d])
      if not int(ID) in excludelist:
        if max(ref[ID][0:(d+1)])>0 and map[ID][d] != 255:
            start = (d>Ndays)*(d-Ndays)
            if map[ID][d] in nodist:
                mapVal=1
                if ref[ID][start:(d+mincount)].count(2) > mincount:
                    refVal=2
                elif ref[ID][start:(d+1)].count(1) > 0:
                    refVal=1
                else:
                    refVal=0
            elif map[ID][d] in dist:
                mapVal=2
                if ref[ID][start:(d+mincount)].count(2) > mincount:
                    refVal=2
                elif ref[ID][start:(d+1)].count(1) > 0:
                    refVal=1
                else:
                    refVal=0
            elif map[ID][d] in old:
                mapVal=2
                if ref[ID][0:(d+mincount)].count(2) > mincount:
                    refVal=2
                else:
                    refVal=1
            else:
                mapVal=0
                refVal=0
            if refVal>0 and mapVal>0:
                p[refVal][mapVal] += 1
                ptotal += 1
            #if refVal != mapVal and refVal>0 and mapVal>0:
            #    print(ID, d, map[ID][d],mapVal, refVal,ref[ID][start:(d+1)])
    if ptotal>0:
      #if p[1][2]/ptotal >0 and p[2][2]==0:#or p[2][1]/ptotal >0
      #  print(ID, strata[ID], "true: ",round(p[2][2]/ptotal,3),"comm: ", round(p[1][2]/ptotal,3), "om: ", round(p[2][1]/ptotal,3))
      with open(name+".txt",'a') as OUT:
        OUT.write(str(ID)+","+str(p[1][1])+","+str(p[1][2])+","+str(p[2][1])+","+str(p[2][2])+","+str(ptotal)+"\n")
      ntotal[strata[ID]] += (p[1][1]+p[1][2]+p[2][1]+p[2][2])/ptotal
      for r in [1,2]:
        for m in [1,2]:
          n[strata[ID]][r][m] += (p[r][m]/ptotal)
    #print(ptotal,end=",")
  if printMatrix:
    print("r1m1,r1m2,r2m1,r2m2")
    for s in range(1,Nstrata):
      for r in (1,2):
        for m in (1,2):
          print(n[s][r][m],end=",")
      print()
  return (n,ntotal)

def accuracy(n, ntotal, strataCounts,name,write=True):
  N = strataCounts
  Nstrata = len(strataCounts)
  Ntotal = sum([N[str(s)] for s in range(1,Nstrata)])

  #Accuracy
  overall = 0
  y = 0
  usersx = 0
  producersx = 0
  OAsub = 0
  area = [[0,0],[0,0]]
  for s in range(1,Nstrata):
    #overall
    overall += ((n[s][1][1] + n[s][2][2])/ntotal[s])*(N[str(s)]/Ntotal)
    oyhmean = (n[s][1][1] + n[s][2][2])/ntotal[s]
    sampvaryhOA = ((n[s][1][1] + n[s][2][2])*((1-oyhmean)**2) + (n[s][1][2] + n[s][2][1])*((0-oyhmean)**2))/(ntotal[s]-1)
    OAsub += N[str(s)]*N[str(s)]*(1-ntotal[s]/N[str(s)])*sampvaryhOA/(ntotal[s])*(1/Ntotal**2)

    #users and producers
    y += (n[s][2][2]/ntotal[s])*N[str(s)]
    usersx += ((n[s][1][2]+n[s][2][2])/ntotal[s])*N[str(s)]
    producersx += ((n[s][2][1]+n[s][2][2])/ntotal[s])*N[str(s)]
    
    #print("strata",s,n[s][1][1],n[s][1][2],n[s][2][1],n[s][2][2])
    for i in range(1,3):
      for j in range(1,3):
        area[i-1][j-1] += (n[s][i][j]/ntotal[s])*(N[str(s)]/Ntotal)
  
  print(area[0][0]*100,area[0][1]*100)
  print(area[1][0]*100,area[1][1]*100)

  overall = overall*100
  overallSE = math.sqrt(OAsub)*100
  if usersx > 0:
    users = (y/usersx)
  else:
    users = "NA"
    usersSE = "NA"
  if producersx > 0:
    producers = (y/producersx)
  else:
    producers = "NA"
    producersSE = "NA"

  UAsub1 = 0
  UAsub2 = 0
  PAsub1 = 0
  PAsub2 = 0
  for s in range(1,Nstrata):
    #users and producers
    yhmean = n[s][2][2]/ntotal[s]
    yhsampvar = ((n[s][2][2])*((1-yhmean)**2) + (n[s][1][1] + n[s][1][2] + n[s][2][1])*((0-yhmean)**2))/(ntotal[s]-1)
    
    if users != "NA":
        if (n[s][1][2]+n[s][2][2]) > 0:
            xuhmean = (n[s][1][2]+n[s][2][2])/ntotal[s]
            xuhsampvar = ((n[s][1][2]+n[s][2][2])*((1-xuhmean)**2) + (n[s][1][1] + n[s][2][1])*((0-xuhmean)**2))/(ntotal[s]-1)
            xyuhsampvar = (n[s][1][1] * (0-yhmean) * (0-xuhmean) + n[s][1][2] * (0-yhmean) * (1-xuhmean) + n[s][2][1] * (0-yhmean) * (0-xuhmean) + n[s][2][2] * (1-yhmean) * (1-xuhmean))/(ntotal[s] - 1)
            UAsub1 += N[str(s)]*xuhmean
            UAsub2 += N[str(s)]**2 * (1 - ntotal[s]/N[str(s)]) * (yhsampvar + (users**2)*xuhsampvar - 2*users*xyuhsampvar)/ntotal[s]
    
    if producers != "NA":
        if (n[s][2][1]+n[s][2][2]) > 0:
            xphmean = (n[s][2][1]+n[s][2][2])/ntotal[s]
            xphsampvar = ((n[s][2][1]+n[s][2][2])*((1-xphmean)**2) + (n[s][1][1] + n[s][1][2])*((0-xphmean)**2))/(ntotal[s]-1)
            xyphsampvar = (n[s][1][1] * (0-yhmean) * (0-xphmean) + n[s][1][2] * (0-yhmean) * (0-xphmean) + n[s][2][1] * (0-yhmean) * (1-xphmean) + n[s][2][2] * (1-yhmean) * (1-xphmean))/(ntotal[s] - 1)
            PAsub1 += N[str(s)]*xphmean
            PAsub2 += N[str(s)]**2 * (1 - ntotal[s]/N[str(s)]) * (yhsampvar + (producers**2)*xphsampvar - 2*producers*xyphsampvar)/ntotal[s]

  
  if users != "NA":
    usersSE = math.sqrt(1/(UAsub1**2) * UAsub2) * 100
    users = users*100
  else:
    usersSE = "NA"
  if producers != "NA":
    producersSE = math.sqrt(1/(PAsub1**2) * PAsub2) * 100
    producers = producers*100
  else:
    producersSE = "NA"
  print("Overall:",overall," +-", overallSE)
  print("Users:",users," +-", usersSE)
  print("Producers:",producers," +-", producersSE)
  if write:
    with open("accuracy.csv",'a') as OUT:
        OUT.write(','.join([name,str(overall),str(overallSE),str(users),str(usersSE),str(producers),str(producersSE)])+"\n")


