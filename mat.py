class Matrix:

    def __init__(self, vals):
        self.vals = vals
        self.rows = len(self.vals)
        self.cols = len(self.vals[0])
        self.dim = (self.rows, self.cols)

    def row(self, r):
        return self.vals[r]
        
    def col(self, c):
        return [r[c] for r in self.vals]

    def mult(self, m):
        res = [
            [0 for _ in range(m.cols)] 
            for _ in range(self.rows)
        ]
        for i, row in enumerate(res):
            for j in range(len(row)):
                res[i][j] = sum([
                    self.vals[i][k] * m.vals[k][j]
                    for k in range(m.rows)
                ])
        return Matrix(res)

    def __str__(self):
        out = []
        for row in self.vals:
            out = [*out, *row]
        return f'{len(self.vals)} {len(self.vals[0])} {" ".join([str(f) for f in out])}'

def matrix_from_list(lst, sz):
    return Matrix([
        lst[i : i + sz]
        for i in range(0, len(lst), sz)
    ])

def read_matrix():
    line = input().strip()
    nums = line.split(' ')
    _, sz = [int(s) for s in nums[:2]]
    lst = [float(s) for s in nums[2:]]
    return matrix_from_list(lst, sz)