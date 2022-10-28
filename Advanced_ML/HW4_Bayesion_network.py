# -*- coding: utf-8 -*-

# DO NOT CHANGE
import pandas as pd
import numpy as np

def get_order(structure): # order
    # structure: dictionary of structure
    #            key is variable and value is parents of a variable
    # return list of learning order of variables 
    # ex) ['A', 'R', 'E', 'S', 'T', 'O']

    order = [k for k, v in structure.items() if v == []] #부모노드가 없는 노드 찾음.
    order.sort()

    while(len(order) < len(structure)):
        for x in range(len(order)):
            pars = order[-(x+1):]
            pars.sort()

            for n, p in structure.items():
                p.sort() # 같은 부모노드를 갖고있을 때, true값을 반환해주기 위해 정렬이 필요
                if pars == p:
                    order.append(n)

    return order
 
    

def learn_parms(data,structure,var_order): #parameter 추정
    # data: training data
    # structure: dictionary of structure
    # var_order: list of learning order of variables
    # return dictionary of trained parameters (key=variable, value=learned parameters)
    dic = {}
    for v in var_order:
        
        #v의 부모노드
        parents = structure[v]
        # parents.sort()
               
        #부모노드 없을때/있을때
        if len(parents) == 0:
            #확률 구해야 함.
            
            df = pd.DataFrame(data[v].value_counts().sort_index()/len(data)).T
            dic[v] = df
            
        else:
             #카테고리들
            category = np.unique(data[v])
            category.sort()
            # 예를들어 !! E의 부모노드가 A,S면 p(E|A,S) = p(E,A,S) / p(A,S)
            p_gb = (data.groupby(parents).size()/len(data)).sort_index() # 분모에 올, 부모노드 확률
            c_gb = (data.groupby(parents+[v]).size()/len(data)).sort_index().unstack() 
            
            for c in category:
                c_gb[c] = c_gb[c]/p_gb
            
            c_gb.columns = category.copy()
            
            dic[v] = c_gb
        dic[v].fillna(0, inplace = True)

    return dic
        
        

                
def print_parms(var_order,parms): #print함수
    # var_order: list of learning order of variables
    # parms: dictionary of trained parameters (key=variable, value=learned parameters)
    # print the trained parameters for each variable
    for var in var_order:
        print('-------------------------')
        print('Variable Name=%s'%(var))
        #TODO: print the trained paramters
        
        if len(parms[var]) == 1:
            print(parms[var].to_string(index=False))
        else:
            print(parms[var])
    
        
    
data=pd.read_csv('https://drive.google.com/uc?export=download&id=1taoE9WlUUN4IbzDzHv7mxk_xSj07f-Zt', sep=' ')

str1={'A':[],'S':[],'E':['A','S'],'O':['E'],'R':['E'],'T':['O','R']} #key : 특정 노드, value: 그 노드의 parents값
order1=get_order(str1)
parms1=learn_parms(data,str1,get_order(str1))
print('-----First Structure------')
print_parms(order1,parms1)
print('')

str2={'A':['E'],'S':['A','E'],'E':['O','R'],'O':['R','T'],'R':['T'],'T':[]}
order2=get_order(str2)
parms2=learn_parms(data,str2,get_order(str2))
print('-----Second Structure-----')
print_parms(order2,parms2)
print('')