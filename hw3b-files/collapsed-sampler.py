import sys
from operator import itemgetter
from random import randint
import random
import numpy as np
import time

alpha=float(sys.argv[6])
beta=float(sys.argv[7])
lmd=float(sys.argv[5])
num_of_iterations=int(sys.argv[8])
burnin=int(sys.argv[9])
outputfile=sys.argv[3]
inputfile_test=sys.argv[2]
inputfile_train=sys.argv[1]
K=int(sys.argv[4])#the number of topics

## create global phi
global_phi=[]
for i in range(K):
	global_phi.append({})


## create collection phi
collection_phi=[]
collection_phi.append([])
collection_phi.append([])


#collection_0_ph=[]
for i in range(K):
	collection_phi[0].append({})
	collection_phi[1].append({})

doc_topics=[]
vocabulary=set()
V=0

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



with open(inputfile_train,'r') as f:
    for line in f:
        tokens=line.strip("\n").split(" ")
        array.append([])
        doc_topics.append([])
        x.append([])
        size_of_doc=len(doc_topics)
        
        for i in range(K):
            doc_topics[size_of_doc-1].append(0)

        for token in tokens:
            vocabulary.add(token)
            random_topic=randint(0,K-1)
            x_d_i=np.random.choice(2,1,p=[1-lmd,lmd])[0]
            x[len(x)-1].append(x_d_i)
            array[len(array)-1].append(random_topic)
            global_NumOftopics[random_topic]+=1
            # construct docs
            doc_topics[size_of_doc-1][random_topic]+=1
            if token in global_phi[random_topic]:
                global_phi[random_topic][token]+=1
            else:
                global_phi[random_topic][token]=1
            if tokens[0]=='1':
            	c_NumOftopics[1][random_topic]+=1
            	if token in collection_phi[1][random_topic]:
            		collection_phi[1][random_topic][token]+=1
            	else:
            		collection_phi[1][random_topic][token]=1
            elif tokens[0]=='0':
            	c_NumOftopics[0][random_topic]+=1
            	if token in collection_phi[0][random_topic]:
            		collection_phi[0][random_topic][token]+=1
            	else:
            		collection_phi[0][random_topic][token]=1
    f.close()

# #print(type(vocabulary))
# V=len(vocabulary)
# print("the number of vocabulary is :",V)
# for i in range(K):
# 	print(global_NumOftopics[i])
# print()
# for i in range(K):
# 	print(c_NumOftopics[0][i])
# 	print(c_NumOftopics[1][i])



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


with open(inputfile_test,"r") as f:
    d=-1
    for line in f:
        d+=1
        tokens=line.strip("\n").split(" ")
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
    f.close()


##store data
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

#print(test_data)





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
            if token in collection_phi[c_index][i]:
            	distribution_i*=(collection_phi[c_index][i][token]+beta)/(sum_nk+V*beta)
            else:
            	distribution_i*=(beta)/(sum_nk+V*beta)
            distributions.append(distribution_i)
    elif flag==0:
        for i in range(K):
            distribution_i=(doc_topics[doc_index][i]+alpha)/(len(array[doc_index])-1+K*alpha)
            sum_nk=global_NumOftopics[i]
            if token in global_phi[i]:
            	distribution_i*=(global_phi[i][token]+beta)/(sum_nk+V*beta)
            else:
            	distribution_i*=beta/(sum_nk+V*beta)
            distributions.append(distribution_i)
    
    return np.array(distributions)/np.sum(distributions)

def smaple_xdi(c_index,doc_index,token,x,z_d_token):
    global vocabulary
    global global_phi
    global collection_phi
    global lmd
    global V
    
    sum_nk=global_NumOftopics[z_d_token]
    p_0=0
    if token in global_phi[z_d_token]:
    	p_0=(1-lmd)*((global_phi[z_d_token][token]+beta)/(sum_nk+V*beta))
    else:
    	p_0=(1-lmd)*(beta/(sum_nk+V*beta))

    sum_nk_c=c_NumOftopics[c_index][z_d_token]
    p_1=0
    if token in collection_phi[c_index][z_d_token]:
    	p_1=lmd*((collection_phi[c_index][z_d_token][token]+beta)/(sum_nk_c+V*beta))
    else:
    	p_1=lmd*(beta/(sum_nk_c+V*beta))
    
    dis= np.array([p_0 / (p_0+p_1), p_1 / (p_0+p_1)])#np.array(p)/np.sum(p)
    return np.random.choice(2,1,p=dis)[0]


def update_dic(z_d_token,token,token_index,doc_index,c_index,flag):
    global vocabulary
    global global_phi
    global collection_phi
    global doc_topics
    global array
    
    original_topic=array[doc_index][token_index]
    global_phi[original_topic][token]-=1
    if token in global_phi[z_d_token]:
    	global_phi[z_d_token][token]+=1
    else:
    	global_phi[z_d_token][token]=1
    collection_phi[c_index][original_topic][token]-=1
    if token in collection_phi[c_index][z_d_token]:
    	collection_phi[c_index][z_d_token][token]+=1
    else:
    	collection_phi[c_index][z_d_token][token]=1
    array[doc_index][token_index]=z_d_token
    doc_topics[doc_index][original_topic]-=1
    doc_topics[doc_index][z_d_token]+=1
    
    global_NumOftopics[original_topic]-=1
    global_NumOftopics[z_d_token]+=1
    c_NumOftopics[c_index][original_topic]-=1
    c_NumOftopics[c_index][z_d_token]+=1  


theta_info=[]
phi_c_info_array=[]
phi_g_info=[]


def generate_phi_theta():
	#theta_info=[]
	global theta_info
	global phi_c_info_array
	global phi_g_phi
	#theta_info=[]
	#phi_g_info=[]
	for i in range(K):
		phi_g_info.append({})
	
	#phi_c_info_array=[]
	phi_c_info_array.append([])
	phi_c_info_array.append([])
	for i in range(K):
		phi_c_info_array[0].append({})
		phi_c_info_array[1].append({})

	for i in range(len(doc_topics)):
		theta_info.append([])
		for j in range(K):
			temp=(doc_topics[i][j]+alpha)/(len(array[i])+K*alpha)
			theta_info[len(theta_info)-1].append(temp)
	
	for i in range(K):
		sum_nk=global_NumOftopics[i] 
		sum_nk_c1=c_NumOftopics[1][i]
		sum_nk_c0=c_NumOftopics[0][i]

		for ele in vocabulary:
			if ele in global_phi[i]:
				phi_g_info[i][ele]=(global_phi[i][ele]+beta)/(sum_nk+V*beta)
			else:
				phi_g_info[i][ele]=beta/(sum_nk+V*beta)
			
			if ele in collection_phi[0][i]:
				phi_c_info_array[0][i][ele]=(collection_phi[0][i][ele]+beta)/(sum_nk_c0+V*beta)
			else:
				phi_c_info_array[0][i][ele]=beta/(sum_nk_c0+V*beta)
			
			if ele in collection_phi[1][i]:
				phi_c_info_array[1][i][ele]=(collection_phi[1][i][ele]+beta)/(sum_nk_c1+V*beta)
			else:
				phi_c_info_array[1][i][ele]=beta/(sum_nk_c1+V*beta)


def output_tofile(output_tofile):
	lines=output_tofile.split("-")
	with open(output_tofile,"a") as f:
		list_vocabulary=list(vocabulary)
		if lines[1]=="phi":
			for ele in vocabulary:
				tempstr=ele
				for i in range(K):
					tempstr+=" {:.13f}".format(store_c_phi[i][ele]/(num_of_iterations-burnin))
				f.write(tempstr)
				f.write("\n")
		elif lines[1]=="phi0":
			for ele in vocabulary:
				tempstr=ele
				for i in range(K):
					tempstr+=" {:.13f}".format(store_c_phi[0][ele]/(num_of_iterations-burnin))
				f.write(tempstr)
				f.write("\n")
		elif lines[1]=='phi1':
			for ele in vocabulary:
				tempstr=ele
				for i in range(K):
					tempstr+=" {:.13f}".format(store_c_phi[1][ele]/(num_of_iterations-burnin))
		elif lines[1]=='theta':
			for i in range(len(store_theta)):
				tempstr="document1 "
				for j in range(K):
					tempstr+=" {:.13f}".format(store_phi[i][j]/(num_of_iterations-burnin))
				f.write(tempstr)
				f.write("\n")
		elif lines[1]=="trainll":
			pass
		elif lines[1]=="testll":
			pass



store_theta=None
#store_test_theta=np.zeros(len(test_theta_info),len(test_theta_info[0]))
store_g_phi=[]
for i in range(K):
	store_g_phi.append({})
store_c_phi=[]
store_c_phi.append([])
store_c_phi.append([])

for i in range(K):
	store_c_phi[0].append({})
	store_c_phi[1].append({})

def burn_in():
	global store_theta
	if store_theta==None:
		store_theta=np.zeros(len(theta_info),len(theta_info[0]))

	for i in range(len(theta_info)):
		for j in range(len(theta_info[0])):
			store_phi[i][j]+=theta_info[i][j]

	for i in range(K):
		for ele in vocabulary:
			if ele in store_g_phi[i]:
				store_g_phi[i][ele]+=phi_g_info[i][ele]
			else:
				store_g_phi[i][ele]=phi_g_info[i][ele]
			if ele in store_c_phi[0][i]:
				store_c_phi[0][i][ele]+=phi_c_info_array[0][i][ele]
			else:
				store_c_phi[0][i][ele]=phi_c_info_array[0][i][ele]

			if ele in store_c_phi[1][i]:
				store_c_phi[1][i][ele]+=phi_c_info_array[1][i][ele]
			else:
				store_c_phi[1][ele]=phi_c_info_array[1][i][ele]



def compute_z_x_testingfile():
    #with open(inputfile_test,"r") as f:
    d=-1
    for tokens in test_data:
    	d+=1
    	#tokens=line.strip("\n").split(" ")
    	i=-1
    	for token in tokens:
    		i+=1
    		flag=x_test[d][i]
    		p_distributions=get_distribution_doc_topic(flag,int(tokens[0]),d,token,doc_topics_test)
    		z_d_token=np.random.choice(K,1,p=p_distributions)[0]
    		x_d_i=smaple_xdi(int(tokens[0]),d,token,x_test,z_d_token)
    		update_test_file_info(x_d_i,d,i,z_d_token)
    #	f.close()    			


def update_test_file_info(x_d_i,doc_index,token_index,z_d_token):
	global array_test
	originaltopic=array_test[doc_index][token_index]
	doc_topics_test[doc_index][originaltopic]-=1
	doc_topics_test[doc_index][z_d_token]+=1
	array_test[doc_index][token_index]=z_d_token
	x_test[doc_index][token_index]=x_d_i


test_theta_info=[]
def compute_test_phi_theta():
	compute_z_x_testingfile()
	for i in range(len(doc_topics_test)):
		test_theta_info.append([])
		for j in range(K):
			temp=(doc_topics_test[i][j]+alpha)/(len(array_test)+K*alpha)
			test_theta_info[len(test_theta_info)-1].append(temp)


def compute_test_log_likelihood():
	pass

def compute_train_log_likelihood(train_data,theta_info,phi_g_info,phi_c_info_array):
	total_likelihood=0
	d=-1
	for tokens in train_data:
		d+=1
		tokens_sum=0
		for token in tokens:
			pre_log_sum=0
			for k in range(K):
				theta_dz=theta_info[d][k]
				phi_z_wdi=phi_g_info[k][token]
				phi_z_wdi_cd=phi_c_info_array[int(tokens[0])][k][token]
				pre_log_sum+=theta_dz*((1-lmd)*phi_z_wdi+lmd*phi_z_wdi_cd)
			tokens_sum+=np.log(pre_log_sum)
		total_likelihood+=tokens_sum
	return total_likelihood




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
    generate_phi_theta()
    if t>=burnin:
    	burn_in()
    #compute_test_phi_theta()
    train_log_likelihood="{:.13f}".format(compute_train_log_likelihood(train_data,theta_info,phi_g_info,phi_c_info_array))
    print("the log likelihood of traing data is:{}".format(train_log_likelihood))
    #test_log_likelihood="{:.13f}".format(compute_train_log_likelihood(test_data,test_theta_info,phi_g_info,phi_c_info_array))
    #print("the log likelihood of testing data is:{}".format(test_log_likelihood))



