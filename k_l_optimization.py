# !pip install -U pymoo

import pymoo
# pymoo.__version__
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.util.plotting import plot
import matplotlib.pyplot as plt



class MyProblem(Problem):
# For minimising SE distances
    def __init__(self):
      super().__init__(n_var=2, n_obj=3, n_ieq_constr = 0, xl=np.array([3, 2]), xu=np.array([10, 4]), vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
      # for SE similarity
      obj1 = -1 * fun1(x[:,0], x[:,1])
      # for SA similarity
      obj2 = fun2(x[:,0], x[:,1])
      # for information loss
      obj3 = fun3(x[:,0], x[:,1])
      # constr = x[:,1] - x[:,0]

      out["F"] = np.column_stack([obj1, obj2, obj3])

      # out["G"] = np.column_stack([constr])


# for SE similarity
def fun1(k, l):
  print("fun1 k = ",k)
  print("l = ",l)
  print(type(k))
  # distances1 = [0.8259, 0.833, 0.8011, 0.8032, 0.7775, 0.7825, 0.7658, 0.7599, 0.8277, 0.8335, 0.8012, 0.8032, 0.7775, 0.7825, 0.7658, 0.7599, 0, 0.8351, 0.802, 0.8034, 0.7777, 0.7825, 0.7658, 0.7599]
  distances1 = [0.8259, 0.833, 0.8011, 0.8032, 0.7775, 0.7825, 0.7658, 0.7599, 0.8277, 0.8335, 0.8012, 0.8032, 0.7775, 0.7825, 0.7658, 0.7599, 0, 0.8351, 0.802, 0.8034, 0.7777, 0.7825, 0.7658, 0.8500]


  m = 0
  dict1 = {}
  for i in range(2, 5):
      for j in range(3, 11):
        dict1[str(j)+','+str(i)] = distances1[m]
        m = m + 1
  print("dict1 : ",dict1)

  out = np.empty(len(k))
  print("out: ",out)

  for i in range(0, len(k)):
    print(str(k[i])+','+str(l[i]))
    out[i] = dict1[str(k[i])+','+str(l[i])]
  print("distances1 : ",distances1)
  print("out : ",out)
  return out

# for SA similarity
def fun2(k, l):
  print("fun2 k = ",k)
  print("l = ",l)
  # distances2 = [0.3658, 0.3512, 0.3534, 0.3513, 0.3586, 0.3478, 0.3474, 0.345, 0.3526, 0.3498, 0.3562, 0.3514, 0.352, 0.3464, 0.3511, 0.3427, 1, 0.3368, 0.3488, 0.3498, 0.3528, 0.3455, 0.3471, 0.3436]
  distances2 = [0.3658, 0.3512, 0.3534, 0.3513, 0.3586, 0.3478, 0.3474, 0.345, 0.3526, 0.3498, 0.3562, 0.3514, 0.352, 0.3464, 0.3511, 0.3427, 1, 0.3368, 0.3488, 0.3498, 0.3528, 0.3455, 0.3471, 0.3400]

  m = 0
  dict1 = {}
  for i in range(2, 5):
      for j in range(3, 11):
        dict1[str(j)+','+str(i)] = distances2[m]
        m = m + 1
  print("dict1 : ",dict1)

  out = np.empty(len(k))
  print("out: ",out)

  for i in range(0, len(k)):
    print(str(k[i])+','+str(l[i]))
    out[i] = dict1[str(k[i])+','+str(l[i])]
  print("distances2 : ",distances2)
  print("out : ",out)
  return out

def fun3(k, l):
  print("fun2 k = ",k)
  print("l = ",l)
  distances2 = [0.2772, 0.2318, 0.2535, 0.2364, 0.2466, 0.2533, 0.2654, 0.2474, 0.2939, 0.2363, 0.2543, 0.2364, 0.2466, 0.2533, 0.2654, 0.2474, 1, 0.2584, 0.2615, 0.2387, 0.2483, 0.2533, 0.2654, 0.2474]

  m = 0
  dict1 = {}
  for i in range(2, 5):
      for j in range(3, 11):
        dict1[str(j)+','+str(i)] = distances2[m]
        m = m + 1
  print("dict1 : ",dict1)

  out = np.empty(len(k))
  print("out: ",out)

  for i in range(0, len(k)):
    print(str(k[i])+','+str(l[i]))
    out[i] = dict1[str(k[i])+','+str(l[i])]
  print("distances2 : ",distances2)
  print("out : ",out)
  return out



problem = MyProblem()


algorithm = NSGA2(pop_size=100,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
               )

res = minimize(problem,
               algorithm,
               termination=('n_gen', 100),
               seed=1,
               save_history=True,
               verbose = False)

print(res.X)
print(res.F)

plot(res.X, no_fill=True)
plot(res.F)

F = res.F
ax = plt.axes(projection ="3d")
ax.scatter3D(F[:, 0], F[:, 1], F[:, 2], color = "blue")
ax.dist = 11.9
ax.set_xlabel('SE(k, l)')
plt.xticks(fontsize=6)
ax.set_ylabel('SA(k, l)')
plt.yticks(fontsize=6)
ax.set_zlabel('AIL')
# ax.set_zticks(np.arange(-3, 4, 1))
ax.tick_params(axis='z', labelsize=6)
# plt.zticks(fontsize=7)
plt.title("Objective Space")
plt.show()