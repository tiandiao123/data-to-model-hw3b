import sys
from operator import itemgetter
from random import randint
import random
import numpy as np


class LDASampler:
	alpha=float(sys.argv[6])
    beta=float(sys.argv[7])
	lmd=float(sys.argv[5])
	num_of_iterations=int(sys.argv[8])
	burnin=int(sys.argv[9])
    outputfile=sys.argv[2]
    inputfile=sys.argv[1]
	
	K=int(sys.argv[4])#the number of topics
    global_phi=[]#global corpose
    collection_phi=[]
    doc_topics=[]
	vocabulary=set()
	V=0 #the number of vocabularies
	array=[]#store every doc's oken's topic assign 


	theta_info=[]
    phi_g_info=[]
    phi_c_info_array=[]

	def __init__(self):
		## create global phi
		for i in range(K):
			global_phi.append({})


		## create collection phi
		collection_phi.append([])
		collection_phi.append([])

        phi_c_info_array.append([])
        phi_c_info_array.append([])

		#collection_0_ph=[]
		for i in range(K):
			collection_phi[0].append({})
			collection_phi[1].append({})

    def initizlization_train(self):
		with open(sys.argv[1],'r') as f:
			for line in f:
				tokens=line.strip("\n").split(" ")
				
				array.append([])
				doc_topics.append([])
				size_of_doc=len(doc_topics)
				for i in range(K):
					doc_topics[size_of_doc-1].append(0)

				for token in tokens:
					vocabulary.add(token)
					random_topic=randint(0,K-1)
					array[len(array)-1].append(random_topic)
					# construct docs
					doc_topics[size_of_doc-1][random_topic]+=1

					if token in global_phi[random_topic]:
						global_phi[random_topic][token]+=1
					else:
						global_phi[random_topic][token]=0

					if tokens[0]=='1':
						if token in collection_phi[1][random_topic]:
							collection_phi[1][random_topic][token]+=1
						else:
							collection_phi[1][random_topic][token]=1
					elif tokens[0]=='0':
						if token in collection_phi[0][random_topic]:
							collection_phi[0][random_topic][token]+=1
						else:
							collection_phi[0][random_topic][token]=1
			V=len(vocabulary)
			f.close()

	def get_distribution_doc_topic(flag,c_index,doc_index,token):
	    distributions=[]
	    if flag==1:
	        sum_nd=0
	        for i in range(K):
	            sum_nd+=doc_topics[doc_index][i]
	        for i in range(K):
	            distribution_i=(doc_topics[doc_index][i]+alpha)/(sum_nd-1+K*alpha)
	            sum_nk=0
	            for ele in vocabulary:
	                if ele in collection_phi[c_index][i]:
	                    sum_nk+=collection_phi[c_index][i][ele]
	            distribution_i*=(collection_phi[c_index][i][token]+beta)/(sum_nk+V*beta)
	            distributions.append(distribution_i)
	    elif flag==0:
	        sum_nd=0
	        for i in range(K):
	            sum_nd+=doc_topics[doc_index][i]
	        for i in range(K):
	            distribution_i=(doc_topics[doc_index][i]+alpha)/(sum_nd-1+K*alpha)
	            sum_nk=0
	            for ele in vocabulary:
	                if ele in global_phi[i]:
	                    sum_nk+=global_phi[i][ele]
	            distribution_i*=(global_phi[i][token]+beta)/(sum_nk+V*beta)
	            distributions.append(distribution_i)
	    
	    return np.array(distributions)/np.sum(distributions)

	def smaple_xdi(c_index,doc_index,token,x,z_d_token):
	    global vocabulary
	    global global_phi
	    global collection_phi
	    global doc_topics
	    global lmd
	    global V
	    
	    sum_nk=0
	    for ele in global_phi[z_d_token]:
	        sum_nk+=global_phi[z_d_token][ele]
	    p_0=(1-lmd)((global_phi[z_d_token][token]+beta)/(sum_nk+V*beta))
	    p_1
	    
	    sum_nk_c=0
	    for ele in collection_phi[c_index][z_d_token]:
	        sum_nk_c+=ele
	    p_1=lmd*((collection_phi[c_index][z_d_token][token]+beta)/(sum_nk_c+V*beta))
	    
	    p=[]
	    p.append(p_0)
	    p.append(p_1)
	    dis=np.array(p)/np.sum(p)
	    return np.random.choice(2,1,p=dis)[0]

	def update_dic(z_d_token,token,token_index,doc_index,c_index,array,flag):

	    original_topic=array[doc_index][token_index]
	    global_phi[original_topic][token]-=1
	    global_phi[z_d_token][token]+=1
	    if flag==1:
	        collection_phi[1][original_topic][token]-=1
	        collection_phi[1][z_d_token][token]+=1
	    elif flag==0:
	        collection_phi[0][original_topic][token]-=1
	        collection_phi[0][z_d_token][token]+=1
	    array[doc_index][token_index]=z_d_token
	    doc_topics[doc_index][original_topic]-=1
	    doc_topics[doc_index][z_d_token]+=1  

	def generate_phi_theta():
		#theta_info=[]
		for i in range(len(doc_topics)):
			theta_info.append([])
			for j in range(K):
				temp=(doc_topics[i][j]+alpha)/(len(array[i])+K*alpha)
				theta_info[len(theta_info)-1].append(temp)
		
		for i in range(K):
			sum_nk=0
			for ele in vocabulary:
				sum_nk+=global_phi[i][ele]
			sum_nk_c1=0
			for ele in vocabulary:
				sum_nk_c1+=collection_phi[1][i][ele]
			sum_nk_c0=0
			for ele in vocabulary:
				sum_nk_c0+=collection_phi[1][i][ele]

			for ele in vocabulary:
				phi_g_info[i][ele]=(global_phi[i][ele]+beta)/(sum_nk+V*beta)
				phi_c_info_array[0][i][ele]=(collection_phi[0][i][ele]+beta)/(sum_nk_c0+V*beta)
				phi_c_info_array[1][i][ele]=(collection_phi[1][i][ele]+beta)/(sum_nk_c1+V*beta)

    #please fill in 
	def burn_in():
		pass
    # please fill in 
	def compute_test_phi_theta():
		pass
    #please fill in
	def compute_test_log_likelihood():
		pass

	def compute_train_log_likelihood():
		total_likelihiid=0
		glbal_topic=[]
		c_topic=[]
		c_topic.append([])
		c_topic.append([])
		for i in range(K):
			temp_sum=0
			temp_sum_c0=0
			temp_sum_c1=0
			for ele in vocabulary:
				if ele in global_phi[i]:
					temp_sum+=global_phi[i][ele]
				if ele in collection_phi[0][i]:
					temp_sum_c0+=collection_phi[0][i][ele]
				if ele in collection_phi[1][i]:
					temp_sum_c1+=collection_phi[1][i][ele]
			global_topic.append(temp_sum)
			c_topic[0].append(temp_sum_c0)
			c_topic[1].append(temp_sum_c1)
		with open(sys.argv[1],'r') as f:
			d=-1
			for line in f:
				d+=1
				tokens=line.strip("\n").split(" ")
				tokens_sum=0
				for token in tokens:
					pre_log_sum=0
					for k in range(K):
						theta_dz=(doc_topics[d][k]+alpha)/(len(array[d])+K*alpha)
						phi_z_wdi=(global_phi[k][token]+beta)/(global_topic[k]+V*beta)
						phi_z_wdi_cd=0
						if int(tokens[0])==1:
							phi_z_wdi_cd=(collection_phi[1][k][token]+beta)/(temp_sum_c1[k]+V*beta)
						elif int(tokens[0])==0:
							phi_z_wdi_cd=(collection_phi[0][k][token]+beta)/(temp_sum_c0[k]+V*beta)
						pre_log_sum+=theta_dz*((1-lmd)*phi_z_wdi+lmd*phi_z_wdi_cd)
					tokens_sum+=np.log(pre_log_sum)
				total_likelihiid+=tokens_sum
			f.close()
		return total_likelihiid

	def gibbs_sampling(num_of_iterations):
		for t in range(num_of_iterations):
		    d=-1
			with open(sys.argv[1],'r') as f:
				for line in f:
		            d+=1
		            tokens=line.strip("\n").split(" ")
		            i=-1
		            for token in tokens:
		                i+=1
		                flag=x[d][i]
		                z_d_token=np.random.choice(K,1,p=get_distribution_doc_topic(flag,int(tokens[0]),d,token))[0]
		                newx=sample_xdi(int(tokens[0],d,token,x,z_d_token)
		                x[d][i]=smaple_xdi(flag,c_index,doc_index,token,x,z_d_token)
		                update_dic(z_d_token,token,i,d,int(tokens[0]),array,flag)

		        f.close()
		    #the following three functions have to be implemented!    
		    generate_phi_theta()
		    if t>=burnin:
		    	burn_in()
		    compute_test_phi_theta()
		    train_log_likelihood="{.13f}".format(compute_train_log_likelihood())
		    print("the log likelihood of traing data is:{}".format(train_log_likelihood))
		    test_log_likelihood="{.13f}".format(compute_test_log_likelihood())
		    print("the log likelihood of testing data is:{}".format(test_log_likelihood)






test=LDASampler()
test.gibbs_sampling(int(sys.argv[8]))