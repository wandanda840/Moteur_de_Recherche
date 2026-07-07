#!/usr/bin/python3

import sys

if(len(sys.argv)!=3):
    print("use: <goldstandart> <hypothesis>\n");
    exit()

GOLD=sys.argv[1]
HYP=sys.argv[2]

ref = {}
listeReq={}
data = open(GOLD,"r").readlines()
for line in data:
    col = line.split()
    idQ=int(col[0])
    idD=int(col[1])

    if(len(col)<2):
        sys.stderr.write("Erreur nombre de colonne ds fichier:"+str(col)+"\n")
        exit()

    if idQ not in ref:
        ref[idQ]= {}
        
    ref[idQ][idD]=1 
    listeReq[idQ]=1
    

hyp = {}
data = open(HYP,"r").readlines()
for line in data:
    col=line.split()
    idQ=int(col[0])
    idD=int(col[1])
    
    if idQ not in hyp:
        hyp[idQ]= {}
    if col[1] not in hyp[idQ]:
        hyp[idQ][idD]=float(col[2])
      
    listeReq[idQ] = 1
    
        

sys.stderr.write("Compute error statistics\n")
sys.stdout.write("__________________________________________________________________________________________________\n")
sys.stdout.write("| Query  | Hypothesis | GoldSantard | Correct | Precision |   Recall  |    F1    |  P@1  |  P@5  |\n")
sys.stdout.write("|--------|------------|-------------|---------|-----------|-----------|----------|-------|-------|\n")
nbReq=0
Ghyp=0
Gref=0
Gtrouve=0
Gpa1=0
Gpa5=0
for req,val in sorted(listeReq.items(), key=lambda t: t[0]):

    #print STDERR "\n$req:\n";
    if req not in hyp:
        continue

    pa5=0
    pa1=0
    rank=1
    trouve=0
    for doc,dist in sorted(hyp[req].items(), key=lambda t: t[1],reverse=True): 
        #sys.stderr.write("{}:{}\n".format(doc,hyp[req][doc]))
        if(req in ref and doc in ref[req]):		
            trouve+=1
            if rank<=1:
                pa1=1
            if rank<=5:                               
                pa5+=1		
        rank+=1
    
    nbhyp=len(hyp[req])
    nbref=0
    if req in ref:
        nbref=len(ref[req])
    P=0
    if(nbhyp>0):
        P=trouve/nbhyp*100.0
    
    R=100
    if nbref>0:
        R=trouve/nbref*100.0
    F=0
    if P+R>0: 
        F=2*(P*R)/(P+R)
    pa1*=100
    if nbref>0:
        pa5*=max(20,100.0/nbref)
    elif nbhyp==0:
        pa5=100
    else:
        pa=0
        
    sys.stdout.write("|  {:3d}   |   {:5d}    |   {:5d}     |   {:5d} | {:6.1f}%   | {:6.1f}%   |  {:6.1f}% |  {:3.0f}% |  {:3.0f}% |\n".format(req,nbhyp,nbref,trouve, P, R , F,pa1,pa5))
    sys.stdout.write("|--------|------------|-------------|---------|-----------|-----------|----------|-------|-------|\n")
    
    Ghyp+=nbhyp
    Gref+=nbref
    Gtrouve+=trouve
    Gpa1+=pa1
    Gpa5+=pa5
    nbReq+=1


P=0
R=0
F=0
if(Ghyp>0):
   P=Gtrouve/Ghyp*100.0
if Gref>0: 
	R=Gtrouve/Gref*100.0
if(P+R>0):
	F=2*(P*R)/(P+R)
Gpa1/=nbReq
Gpa5/=nbReq
sys.stdout.write("| overall|   {:5d}    |   {:5d}     |   {:5d} | {:6.1f}%   | {:6.1f}%   |  {:6.1f}% |  {:3.0f}% |  {:3.0f}% |\n".format(Ghyp,Gref,Gtrouve, P, R , F,Gpa1,Gpa5))
sys.stdout.write("|--------|------------|-------------|---------|-----------|-----------|----------|-------|-------|\n")
