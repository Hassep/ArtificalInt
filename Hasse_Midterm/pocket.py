import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.plotly as py
import os, subprocess
import copy
from numpy import array
from numpy import genfromtxt

class Pocket:
    def __init__(self, N):
        # Random linearly separated data
        # xA,yA,xB,yB,xC,yC,xD,yD,xE,yE,xF,yF,xG,yG,xH,yH,xI,yI,xJ,yJ = [random.uniform(-1, 1) for i in range(20)]
        #self.V = (np.random.rand(3)*2)-1
        # self.V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
        self.X = self.generate_points(N)
 
    def generate_points(self, N):
        X, y = self.make_dataset()
        bX = []
        #aX = []
        for k in range(0,N) :
            bX.append((np.concatenate(([1], X[k,:])), y[k]))
        #A = np.ones((N,1))
        #print A
        print X
        # # this will calculate linear regression at this point

        # for i in range(0,N):
        #     aX.append((np.concatenate(([1], X[i,:]))))
        # aX = array(aX)
        # print aX
        #aX = np.concatenate((A, X), axis=1) # adds the 1 constant
        X = np.concatenate((np.ones((N,1)), X),axis=1)
        self.linRegW = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)   # lin reg

        return bX

    def make_dataset(self):
        # read digits data & split it into X and y
        dataset = genfromtxt('features.csv', delimiter=' ')

        y = dataset[:, 0]
        X = dataset[:, 1:]

        y[y<>0] = -1
        y[y==0] = +1

        #plots data
        c0 = plt.scatter(X[y==-1,0], X[y==-1,1], s=20, color='r', marker='x')
        c1 = plt.scatter(X[y==1,0], X[y==1,1], s=20, color='b', marker='o')

        # displays legend
        plt.legend((c0, c1), ('All Other Numbers -1', 'Number Zero +1'), loc='upper right', scatterpoints=1, fontsize=11)

        #displays axis legends and title
        plt.xlabel(r'$x_1$')
        plt.xlabel(r'$x_2$')
        plt.title(r'Intensity and Symmetry of Digits')

        #saves the figure into a .pdf files
        plt.savefig('midterm.plot.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

        return X, y
    
 
    def plot(self, mispts=None, vec=None, save=False):
        fig = plt.figure(figsize=(5,5))
        plt.xlim(0,.7)
        plt.ylim(-8,1)
        l = np.linspace(0,.7)
        V = self.linRegW
        V = self.bestW
        a, b = -V[1]/V[2], -V[0]/V[2]
        l = np.linspace(0,.7)
        plt.plot(l, a*l+b, 'k-')
        cols = {1: 'r', -1: 'b'}
        for x,s in self.X:
            plt.plot(x[1], x[2], cols[s]+'o')
        if mispts:
            for x,s in mispts:
                plt.plot(x[1], x[2], cols[s]+'.')
        if vec != None:
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)
        if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points' \
                          % (str(len(self.X)),str(len(mispts))))
            plt.savefig('p_N%s' % (str(len(self.X))), \
                        dpi=200, bbox_inches='tight') 

    def classification_error(self, vec, pts=None):
        # Error defined as fraction of misclassified points
        if not pts:
            pts = self.X
        M = len(pts)
        n_mispts = 0
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                n_mispts += 1
        error = n_mispts / float(M)
        return error
 
    def choose_miscl_point(self, vec):
        # Choose a random point among the misclassified
        pts = self.X
        mispts = []
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0,len(mispts))]
 
    def pla(self, save=False):
        # Initialize the weigths to zeros
        w = np.zeros(3)
        self.bestW = copy.deepcopy(w)
        self.plaError=[]
        self.pocketError=[]

        X, N = self.X, len(self.X)
        it = 0

        self.plaError.append(self.classification_error(w))
        self.pocketError.append(self.plaError[it])

        # Iterate until all points are correctly classified
        while (self.plaError[it] != 0 and it!=5000):
            it += 1
            # Pick random misclassified point
            x, s = self.choose_miscl_point(w)
            # Update weights
            w += s*x
            self.plaError.append(self.classification_error(w))
            if (self.pocketError[it-1] > self.plaError[it]):
                self.pocketError.append(self.plaError[it])
                self.bestW = copy.deepcopy(w)
            else:
                self.pocketError.append(self.pocketError[it-1])
            if save:
                self.plot(vec=w)
                plt.title('N = %s, Iteration %s\n' \
                          % (str(N),str(it)))
                plt.savefig('p_N%s_it%s' % (str(N),str(it)), \
                            dpi=200, bbox_inches='tight')

                plt.close()
                plt.clf()
            print self.pocketError[it]
            #print self.bestW
            print it

        self.w = w
        print it
        return it
 
    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)

def main():
    ds = genfromtxt('features.csv', delimiter=' ')
    row_count = sum(1 for row in ds)
    p = Pocket(row_count)
    p.pla(save=False)

main()