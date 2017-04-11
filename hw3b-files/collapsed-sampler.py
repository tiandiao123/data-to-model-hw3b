import sys
from operator import itemgetter
from random import randint
import random
import math
#import numpy as np
import time
from decimal import *

inputfile_test=sys.argv[2]
inputfile_train=sys.argv[1]

outputfile=[]
index=3
while sys.argv[index].startswith("output"):
    outputfile.append(sys.argv[index])
    index+=1

K=int(sys.argv[index])
index+=1
lmd=float(sys.argv[index])
index+=1
alpha=float(sys.argv[index])
index+=1
beta=float(sys.argv[index])
index+=1
num_of_iterations=int(sys.argv[index])
index+=1
burnin=int(sys.argv[index])

print(burnin)

data=None
with open(inputfile_train,'r') as f:
	data=f.readlines()
	f.close()
train_data=[]
for line in data:
	tokens=line.strip("\n").split(" ")
	train_data.append(tokens)


test_data1=None
with open(inputfile_test,'r') as f:
	test_data1=f.readlines()
	f.close()
test_data=[]
for line in test_data1:
	tokens=line.strip("\n").split(" ")
	test_data.append(tokens)
 

vocabulary_train=set()
vocabulary_test=set()
vocabulary=set()
for tokens in train_data:
    for token in tokens:
        vocabulary.add(token)
        vocabulary_train.add(token)
V_train=len(vocabulary_train)

for tokens in test_data:
    for token in tokens:
        vocabulary_test.add(token)
        vocabulary.add(token)
        
list1=list(vocabulary)
token2index={}
for i in range(len(list1)):
    token2index[list1[i]]=i

V_test=len(vocabulary_test)
V=len(vocabulary)




global_phi=[]
for i in range(V):
    global_phi.append([])
    for j in range(K):
        global_phi[i].append(0)
        


## create collection phi
collection_phi=[]
collection_phi.append([])
collection_phi.append([])


#collection_0_ph=[]
for i in range(V):
    collection_phi[0].append([])
    collection_phi[1].append([])
    for j in range(K):
        collection_phi[0][i].append(0)
        collection_phi[1][i].append(0)

doc_topics=[]

array=[]
x=[]
global_NumOftopics=[]
for i in range(K):
	global_NumOftopics.append(0)

c_NumOftopics=[]
c_NumOftopics.append([])
c_NumOftopics.append([])
for i in range(K):
	c_NumOftopics[0].append(0)
	c_NumOftopics[1].append(0)
    
    
for tokens in train_data:
    #tokens=line.strip("\n").split(" ")
    array.append([])
    doc_topics.append([])
    x.append([])
    size_of_doc=len(doc_topics)
    for i in range(K):
        doc_topics[size_of_doc-1].append(0)
    for token in tokens:
        #vocabulary.add(token)
        random_topic=randint(0,K-1)

        randomnumber=random.uniform(0,1)
        x_d_i=0
        if randomnumber<=1-lmd:
            x_d_i=0
        else:
            x_d_i=1
        #x_d_i=np.random.choice(2,1,p=[1-lmd,lmd])[0]
        x[len(x)-1].append(x_d_i)
        array[len(array)-1].append(random_topic)
        global_NumOftopics[random_topic]+=1
        # construct docs
        doc_topics[size_of_doc-1][random_topic]+=1
        global_phi[token2index[token]][random_topic]+=1
        
        c_NumOftopics[int(tokens[0])][random_topic]+=1
        collection_phi[int(tokens[0])][token2index[token]][random_topic]+=1
        
        
x_test=[]
array_test=[]
doc_topics_test=[]
global_NumOftopics_test=[]
for i in range(K):
	global_NumOftopics_test.append(0)

c_NumOftopics_test=[]
c_NumOftopics_test.append([])
c_NumOftopics_test.append([])
for i in range(K):
	c_NumOftopics_test[0].append(0)
	c_NumOftopics_test[1].append(0)
    
    
d=-1
for tokens in test_data:
    d+=1
    array_test.append([])
    x_test.append([])
    doc_topics_test.append([])
    for i in range(K):
        doc_topics_test[len(doc_topics_test)-1].append(0)
        
    for i in range(len(tokens)):
        random_topic=randint(0,K-1)
        randomnumber=random.uniform(0,1)
        x_d_i=0
        if randomnumber<=1-lmd:
            x_d_i=0
        else:
            x_d_i=1


        global_NumOftopics_test[random_topic]+=1
        c_NumOftopics_test[int(tokens[0])][random_topic]+=1

        doc_topics_test[len(doc_topics_test)-1][random_topic]+=1
        x_test[len(x_test)-1].append(x_d_i)
        array_test[d].append(random_topic)

print len(doc_topics_test),len(doc_topics_test[0]),len(test_data)
        
def get_distribution_doc_topic(flag,c_index,doc_index,token,token_index,doc_topics,istest,array):
    global K
    global alpha
    global beta
    global vocabulary
    global global_phi
    global collection_phi
    global doc_topics_test
    global array_test

    if istest==False:
        original_topic=array[doc_index][token_index]
        global_phi[token2index[token]][original_topic]-=1
        collection_phi[c_index][token2index[token]][original_topic]-=1
        doc_topics[doc_index][original_topic]-=1
        global_NumOftopics[original_topic]-=1
        c_NumOftopics[c_index][original_topic]-=1
    else:
        original_topic=array[doc_index][token_index]
        doc_topics[doc_index][original_topic]-=1


    
    distributions=[]
    if flag==1:
        for i in range(K):
            distribution_i=(doc_topics[doc_index][i]+alpha)/float(len(array[doc_index])-1+K*alpha)
            sum_nk=c_NumOftopics[c_index][i]
            distribution_i*=(collection_phi[c_index][token2index[token]][i]+beta)/float(sum_nk+V_train*beta)
            distributions.append(distribution_i)
    elif flag==0:
        for i in range(K):
            distribution_i=(doc_topics[doc_index][i]+alpha)/float(len(array[doc_index])-1+K*alpha)
            sum_nk=global_NumOftopics[i]
            distribution_i*=(global_phi[token2index[token]][i]+beta)/float(sum_nk+V_train*beta)
            distributions.append(distribution_i)
    randomnumber=random.uniform(0,1)
    sumofd=sum(distributions)
    p=[]
    for i in range(K):
        if i==0:
            p.append(distributions[i]/float(sumofd))
        else:
            p.append(p[i-1]+distributions[i]/float(sumofd))
    for i in range(K):
        if i==0 and randomnumber<p[0]:
            return 0
        if i>=1:
            if p[i-1]<=randomnumber and p[i]>randomnumber:
                return i
    return K-1



def smaple_xdi(c_index,doc_index,token,x,z_d_token):
    global vocabulary
    global global_phi
    global collection_phi
    global lmd
    global V
    
    sum_nk=global_NumOftopics[z_d_token]
    p_0=(1-lmd)*(float(global_phi[token2index[token]][z_d_token]+beta)/(sum_nk+V_train*beta))

    sum_nk_c=c_NumOftopics[c_index][z_d_token]
    p_1=lmd*(float(collection_phi[c_index][token2index[token]][z_d_token]+beta)/(sum_nk_c+V_train*beta))
    
    randomnumber=random.uniform(0,1)

    sumofp=sum([p_0,p_1])
    p=p_0/sumofp

    if randomnumber<p:
        return 0
    else:
        return 1


def update_dic(z_d_token,token,token_index,doc_index,c_index,flag):
    global vocabulary
    global global_phi
    global collection_phi
    global doc_topics
    global array
    
    
    global_phi[token2index[token]][z_d_token]+=1
    
    collection_phi[c_index][token2index[token]][z_d_token]+=1
    array[doc_index][token_index]=z_d_token
    
    doc_topics[doc_index][z_d_token]+=1
    global_NumOftopics[z_d_token]+=1
    c_NumOftopics[c_index][z_d_token]+=1  


theta_info=[]
for i in range(len(doc_topics)):
    theta_info.append([])
    for j in range(K):
        theta_info[i].append(0)

phi_c_info_array=[]
phi_g_info=[]

for i in range(V):
    phi_g_info.append([])
    for j in range(K):
        phi_g_info[i].append(0)
    
    #phi_c_info_array=[]
phi_c_info_array.append([])
phi_c_info_array.append([])
for i in range(V):
    phi_c_info_array[0].append([])
    phi_c_info_array[1].append([])
    for j in range(K):
        phi_c_info_array[0][i].append(0)
        phi_c_info_array[1][i].append(0)

def generate_phi_theta():
    global theta_info
    global phi_c_info_array
    global collection_phi
    global global_phi
    global vocabulary
    global global_NumOftopics
    global c_NumOftopics

    for i in range(len(doc_topics)):
        for j in range(K):
            temp=(doc_topics[i][j]+alpha)/(len(array[i])+K*alpha)
            theta_info[i][j]=temp
    
    for i in range(K):
        sum_nk=global_NumOftopics[i] 
        sum_nk_c1=c_NumOftopics[1][i]
        sum_nk_c0=c_NumOftopics[0][i]

        for ele in vocabulary:
            phi_g_info[token2index[ele]][i]=(global_phi[token2index[ele]][i]+beta)/(sum_nk+V_train*beta)
            phi_c_info_array[0][token2index[ele]][i]=(collection_phi[0][token2index[token]][i]+beta)/(sum_nk_c0+V_train*beta)
            phi_c_info_array[1][token2index[ele]][i]=(collection_phi[1][token2index[token]][i]+beta)/(sum_nk_c1+V_train*beta)


def compute_train_log_likelihood(data,theta_info,phi_g_info,phi_c_info_array):
    total_likelihood=0
    d=-1
    for tokens in data:
        d+=1
        tokens_sum=0
        for token in tokens:
            pre_log_sum=0
            for k in range(K):
                theta_dz=theta_info[d][k]
                phi_z_wdi=phi_g_info[token2index[token]][k]
                phi_z_wdi_cd=phi_c_info_array[int(tokens[0])][token2index[token]][k]
                pre_log_sum+=theta_dz*((1-lmd)*phi_z_wdi+lmd*phi_z_wdi_cd)
            tokens_sum+=math.log(pre_log_sum)
        total_likelihood+=tokens_sum
    return total_likelihood

def compute_z_x_testingfile():
    global array_test
    d=-1
    for tokens in test_data:
        d+=1
        i=-1
        for token in tokens:
            i+=1
            flag=x_test[d][i]
            z_d_token=get_distribution_doc_topic(flag,int(tokens[0]),d,token,i,doc_topics_test,True,array_test)
            x_d_i=smaple_xdi(int(tokens[0]),d,token,x_test,z_d_token)
            update_test_file_info(x_d_i,d,i,z_d_token)           


def update_test_file_info(x_d_i,doc_index,token_index,z_d_token):
    global array_test
    #originaltopic=array_test[doc_index][token_index]
    #doc_topics_test[doc_index][originaltopic]-=1
    doc_topics_test[doc_index][z_d_token]+=1
    array_test[doc_index][token_index]=z_d_token
    x_test[doc_index][token_index]=x_d_i


test_theta_info=[]
for i in range(len(doc_topics_test)):
    test_theta_info.append([])
    for j in range(K):
        test_theta_info[i].append(0)

def compute_test_phi_theta():
    compute_z_x_testingfile()
    global test_theta_info
    for i in range(len(doc_topics_test)):
        for j in range(K):
            temp=(doc_topics_test[i][j]+alpha)/(len(array_test[i])+K*alpha)
            test_theta_info[i][j]=temp


store_theta=[]
for i in range(len(doc_topics)):
    store_theta.append([])
    for j in range(K):
        store_theta[i].append(0)


#store_test_theta=np.zeros(len(test_theta_info),len(test_theta_info[0]))
store_g_phi=[]
for i in range(V):
    store_g_phi.append([])
    for j in range(K):
        store_g_phi[i].append(0)


store_test_theta=[]
for i in range(len(doc_topics_test)):
    store_test_theta.append([])
    for j in range(K):
        store_test_theta[i].append(0)

store_c_phi=[]
store_c_phi.append([])
store_c_phi.append([])

for i in range(V):
    store_c_phi[0].append([])
    store_c_phi[1].append([])
    for j in range(K):
        store_c_phi[0][i].append(0)
        store_c_phi[1][i].append(0)


def burn_in():
    global store_theta
    global store_g_phi
    global store_c_phi

    for i in range(len(doc_topics)):
        for j in range(K):
            store_theta[i][j]+=theta_info[i][j]

    for i in range(len(doc_topics_test)):
        for j in range(K):
            store_test_theta[i][j]+=test_theta_info[i][j]

    for i in range(K):
        for ele in vocabulary:
            store_g_phi[token2index[ele]][i]+=phi_g_info[token2index[ele]][i]
            store_c_phi[0][token2index[ele]][i]+=phi_c_info_array[0][token2index[ele]][i]
            store_c_phi[1][token2index[ele]][i]+=phi_c_info_array[1][token2index[ele]][i]



def output_tofile(outputtofile):
    global vocabulary
    global vocabulary_train
    global vocabulary_test

    lines=outputtofile.split("-")
    with open(outputtofile,"w+") as f:
        #list_vocabulary=list(vocabulary)
        if lines[1]=="phi":
            for ele in vocabulary_train:
                tempstr=ele
                for i in range(K):
                    tempstr+=" {:.13f}".format(store_g_phi[token2index[ele]][i])
                f.write(tempstr)
                f.write("\n")
        elif lines[1]=="phi0":
            for ele in vocabulary_train:
                tempstr=ele
                for i in range(K):
                    tempstr+=" {:.13f}".format(store_c_phi[0][token2index[ele]][i])
                f.write(tempstr)
                f.write("\n")
        elif lines[1]=="phi1":
            for ele in vocabulary:
                tempstr=ele
                for i in range(K):
                    tempstr+=" {:.13f}".format(store_c_phi[1][token2index[ele]][i])
                f.write(tempstr)
                f.write("\n")
        elif lines[1]=='theta':
            for i in range(len(store_theta)):
                tempstr="document "+str(i)+" :"
                for j in range(K):
                    tempstr+=" {:.13f}".format(store_theta[i][j])
                f.write(tempstr)
                f.write("\n")
        elif lines[1]=="trainll":
            for i in range(len(log_likelihood_train)):
                temp=str(log_likelihood_train[i])
                f.write(temp)
                f.write("\n")
        elif lines[1]=="testll":
            for i in range(len(log_likeihood_test)):
                temp=str(log_likeihood_test[i])
                f.write(temp)
                f.write("\n")
        f.close()



log_likelihood_train=[]
log_likeihood_test=[]

for t in range(num_of_iterations):
    d=-1
    start=time.time()
    print "iteration:",t
    for tokens in train_data:
    	d+=1
    	i=-1
    	for token in tokens:
    		i+=1
    		flag=x[d][i]
    		z_d_token=get_distribution_doc_topic(flag,int(tokens[0]),d,token,i,doc_topics,False,array)
    		#z_d_token=np.random.choice(K,1,p=get_distribution_doc_topic(flag,int(tokens[0]),d,token,doc_topics))[0]
    		x[d][i]=smaple_xdi(int(tokens[0]),d,token,x,z_d_token)
    		update_dic(z_d_token,token,i,d,int(tokens[0]),flag)
    
    generate_phi_theta()
    if t>=burnin:
        burn_in()
    compute_test_phi_theta()
    log_likelihood_train.append(compute_train_log_likelihood(train_data,theta_info,phi_g_info,phi_c_info_array))
    train_log_likelihood="{:.13f}".format(log_likelihood_train[len(log_likelihood_train)-1])
    print("the log likelihood of traing data is:{}".format(train_log_likelihood))

    log_likeihood_test.append(compute_train_log_likelihood(test_data,test_theta_info,phi_g_info,phi_c_info_array))
    test_log_likelihood="{:.13f}".format(log_likeihood_test[len(log_likeihood_test)-1])
    print("the log likelihood of testing data is:{}".format(test_log_likelihood))
    end=time.time()
    print end-start
    print "iteration ",t," ends"



for i in range(len(doc_topics)):
    for j in range(K):
        store_theta[i][j]/=float(num_of_iterations-burnin)

for i in range(len(doc_topics_test)):
    for j in range(K):
        store_test_theta[i][j]/=float(num_of_iterations-burnin)

for i in range(K):
    for ele in vocabulary:
        store_g_phi[token2index[ele]][i]/=float(num_of_iterations-burnin)
        store_c_phi[0][token2index[ele]][i]/=float(num_of_iterations-burnin)
        store_c_phi[1][token2index[ele]][i]/=float(num_of_iterations-burnin)



output_tofile("output.txt-trainll")
output_tofile("output.txt-phi")
output_tofile("output.txt-phi0")
output_tofile("output.txt-phi1")
output_tofile("output.txt-theta")
output_tofile("output.txt-testll")

test_average_log_likelihood=compute_train_log_likelihood(test_data,store_test_theta,store_g_phi,store_c_phi)
res="{:.13f}".format(test_average_log_likelihood)
print("the test average likelihood is:{}".format(res))
