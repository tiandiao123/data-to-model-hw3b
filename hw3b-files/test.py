import sys
from operator import itemgetter
from random import randint
import random
import numpy as np
import time

alpha=0.1
beta=0.01
lmd=0.5
num_of_iterations=10
burnin=8
outputfile="output.txt"
inputfile_test="input-test.txt"
inputfile_train="input-train.txt"
K=10#the number of topics


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
V_train=len(vocabulary)

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
        x_d_i=np.random.choice(2,1,p=[1-lmd,lmd])[0]
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
    #tokens=line.strip("\n").split(" ")
    array_test.append([])
    x_test.append([])
    doc_topics_test.append([])
    for i in range(K):
        doc_topics_test[len(doc_topics_test)-1].append(0)
        
    for i in range(len(tokens)):
        random_topic=randint(0,K-1)
        x_d_i=np.random.choice(2,1,p=[1-lmd,lmd])[0]

        global_NumOftopics_test[random_topic]+=1
        c_NumOftopics_test[int(tokens[0])][random_topic]+=1

        doc_topics_test[len(doc_topics_test)-1][random_topic]+=1
        x_test[len(x_test)-1].append(x_d_i)
        array_test[d].append(random_topic)
        
def get_distribution_doc_topic(flag,c_index,doc_index,token,doc_topics):
    global K
    global alpha
    global beta
    global vocabulary
    global global_phi
    global collection_phi
    
    distributions=[]
    if flag==1:
        for i in range(K):
            distribution_i=(doc_topics[doc_index][i]+alpha)/(len(array[doc_index])-1+K*alpha)
            sum_nk=c_NumOftopics[c_index][i]
            distribution_i*=(collection_phi[c_index][token2index[token]][i]+beta)/(sum_nk+V_train*beta)
#             if token in collection_phi[c_index][i]:
#             	distribution_i*=(collection_phi[c_index][i][token]+beta)/(sum_nk+V*beta)
#             else:
#             	distribution_i*=(beta)/(sum_nk+V*beta)
            distributions.append(distribution_i)
    elif flag==0:
        for i in range(K):
            distribution_i=(doc_topics[doc_index][i]+alpha)/(len(array[doc_index])-1+K*alpha)
            sum_nk=global_NumOftopics[i]
            distribution_i*=(global_phi[token2index[token]][i]+beta)/(sum_nk+V_train*beta)
#             if token in global_phi[i]:
#             	distribution_i*=(global_phi[i][token]+beta)/(sum_nk+V*beta)
#             else:
#             	distribution_i*=beta/(sum_nk+V*beta)
            distributions.append(distribution_i)
    
    return np.array(distributions)/np.sum(distributions)






def smaple_xdi(c_index,doc_index,token,x,z_d_token):
    global vocabulary
    global global_phi
    global collection_phi
    global lmd
    global V
    
    sum_nk=global_NumOftopics[z_d_token]
    p_0=(1-lmd)*((global_phi[token2index[token]][z_d_token]+beta)/(sum_nk+V_train*beta))
#     if token in global_phi[z_d_token]:
#     	p_0=(1-lmd)*((global_phi[z_d_token][token]+beta)/(sum_nk+V*beta))
#     else:
#     	p_0=(1-lmd)*(beta/(sum_nk+V*beta))

    sum_nk_c=c_NumOftopics[c_index][z_d_token]
    p_1=lmd*((collection_phi[c_index][token2index[token]][z_d_token]+beta)/(sum_nk_c+V_train*beta))
#     if token in collection_phi[c_index][z_d_token]:
#     	p_1=lmd*((collection_phi[c_index][z_d_token][token]+beta)/(sum_nk_c+V*beta))
#     else:
#     	p_1=lmd*(beta/(sum_nk_c+V*beta))
    
    dis= np.array([p_0 / (p_0+p_1), p_1 / (p_0+p_1)])#np.array(p)/np.sum(p)
    return np.random.choice(2,1,p=dis)[0]






def update_dic(z_d_token,token,token_index,doc_index,c_index,flag):
    global vocabulary
    global global_phi
    global collection_phi
    global doc_topics
    global array
    
    original_topic=array[doc_index][token_index]
    global_phi[token2index[token]][original_topic]-=1
    global_phi[token2index[token]][z_d_token]+=1
#     if token in global_phi[z_d_token]:
#     	global_phi[z_d_token][token]+=1
#     else:
#     	global_phi[z_d_token][token]=1
    collection_phi[c_index][token2index[token]][original_topic]-=1
    collection_phi[c_index][token2index[token]][z_d_token]+=1
#     if token in collection_phi[c_index][z_d_token]:
#     	collection_phi[c_index][z_d_token][token]+=1
#     else:
#     	collection_phi[c_index][z_d_token][token]=1
    array[doc_index][token_index]=z_d_token
    doc_topics[doc_index][original_topic]-=1
    doc_topics[doc_index][z_d_token]+=1
    
    global_NumOftopics[original_topic]-=1
    global_NumOftopics[z_d_token]+=1
    c_NumOftopics[c_index][original_topic]-=1
    c_NumOftopics[c_index][z_d_token]+=1  
    
    
for t in range(num_of_iterations):
    d=-1
    start=time.time()
    print("iteration:",t)
    for tokens in train_data:
    	d+=1
    	i=-1
    	for token in tokens:
    		i+=1
    		flag=x[d][i]
    		#t0 = time.time()
    		z_d_token=np.random.choice(K,1,p=get_distribution_doc_topic(flag,int(tokens[0]),d,token,doc_topics))[0]
    		#t1 = time.time()
    		x[d][i]=smaple_xdi(int(tokens[0]),d,token,x,z_d_token)
    		#t2 = time.time()
    		update_dic(z_d_token,token,i,d,int(tokens[0]),flag)
    		#t3 = time.time()
    		#print ("get_distribution ", (t1-t0))
    		#print ("sample_xdi ", (t2-t1))
    		#print ("update_dic ", (t3-t2))
    end=time.time()
    print(end-start)
    print("begin compute phi and theta")
    
    #generate_phi_theta()
    #if t>=burnin:
    #	burn_in()
    #compute_test_phi_theta()
    #train_log_likelihood="{:.13f}".format(compute_train_log_likelihood(train_data,theta_info,phi_g_info,phi_c_info_array))
    #print("the log likelihood of traing data is:{}".format(train_log_likelihood))
    #test_log_likelihood="{:.13f}".format(compute_train_log_likelihood(test_data,test_theta_info,phi_g_info,phi_c_info_array))
    #print("the log likelihood of testing data is:{}".format(test_log_likelihood))
    
    
 
