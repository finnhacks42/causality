
class Test():
    def __init__(self):
        self.parents = 1

    def parents(self,values):
        return values

t = Test()

def math_condition(left,*right):
    result = "P("+left
    first = True
    for cond in right:
        print(cond)
        if len(cond) > 0:
            if first:
                result+="|"
                first = False
            else:
                result+=","
            result+=",".join(cond)
    result +=")"
    return result




print(math_condition('x',[],['y'],['z','w']))
            
