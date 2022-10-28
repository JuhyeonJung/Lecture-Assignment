def func1(list_):
    print(min(list_))
    
def func2(list_):
    print(sorted(list_,reverse=True))
    
def func3(list_):
    print('plus',max(list_) + min(list_))
    print('minus',max(list_) - min(list_))

def func4(list_):
    try:
        print('몫',max(list_)//min(list_))
        print('나머지',max(list_)%min(list_))
    except ZeroDivisionError:
        print('zero division error!!')
        list_.remove(0)
        print('몫',max(list_)//min(list_))
        print('나머지',max(list_)%min(list_))
        
        