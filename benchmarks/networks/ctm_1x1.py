class TN_CTM_1X1():
    r"""
    # C1--(1)1 1(0)----T1--(3)9 9(0)----C2
    # 0(0)            (1,2)             8(1)
    # 0(0)        100  2  5             8(0)
    # |              \ 2  5              |
    # T4--(2)3 3-------a--|---10 10(1)---T2
    # |                |  |              |
    # |   (3)6 6----------a*--11 11(2)   |
    # 14(1)           15 16 \101        17(3)
    # 14(0)           (0,1)             17(0)
    # C4--(1)12 12(2)--T3--(3)13 13(1)--C3
    """
    idx_dims= { "D": [2,3,16,10, 5,6,15,11],
                "X": [0,14, 12,13, 17,8, 9,1],
                "p": [100,101] }

    @staticmethod
    def _gen_contract_tn(I,I_out): 
        return "C1",[0,1],"T1",[1,2,5,9],"T4",[0,14,3,6], \
        "C2",[9,8],"T2",[8,10,11,17],"C3",[17,13],"T3",[15,16,12,13],"C4",[14,12], \
        "a",[I[0],2,3,15,10],"a.conj()",[I[1],5,6,16,11],I_out 
    
    @classmethod
    def build_network(cls, D= 4, X= 64, p= 2, open_idx=[]):
        I= sum([[100+2*x,100+2*x+1] if x in open_idx else [100+2*x]*2 for x in [0,]],[])
        I_out= [100+2*x for x in open_idx]+[100+2*x+1 for x in open_idx]

        dims= {"D": D, "X": X, "p": p}
        tn= cls._gen_contract_tn(I,I_out)

        inputs = [tuple(tn[i]) for i in range(1,len(tn)-1,2)] 
        output = tuple(tn[-1])
        size_dict = { idx: dims[k] for k in cls.idx_dims for idx in cls.idx_dims[k] }

        return inputs, output, size_dict