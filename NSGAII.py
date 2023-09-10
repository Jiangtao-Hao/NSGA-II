#  -*-  codeing  =  utf-8  -*-
#  @Time  :2022/11/18  22:41
#  @Author: 郝江涛
#  @File  :  NSGAII.py
#  @Software:  PyCharm
from collections import defaultdict
import numpy as np
import random
import matplotlib.pyplot as plt
import math


class Individual(object):
    def __init__(self):
        self.solution = None  # 实际赋值中是一个 nparray 类型，方便进行四则运算
        self.objective = defaultdict()

        self.n = 0  # 解p被几个解所支配，是一个数值（左下部分点的个数）
        self.rank = 0  # 解所在第几层
        self.S = []  # 解p支配哪些解，是一个解集合（右上部分点的内容）
        self.distance = 0  # 拥挤度距离

    def bound_process(self, bound_min, bound_max):
        """
        对解向量 solution 中的每个分量进行定义域判断，超过最大值，将其赋值为最大值；小于最小值，赋值为最小值
        :param bound_min: 定义域下限
        :param bound_max:定义域上限
        :return:
        """
        for i, item in enumerate(self.solution):
            if item > bound_max:
                self.solution[i] = bound_max
            elif item < bound_min:
                self.solution[i] = bound_min

    def calculate_objective(self, objective_fun):
        self.objective = objective_fun(self.solution)

    # 重载小于号“<”
    def __lt__(self, other):
        v1 = list(self.objective.values())
        v2 = list(other.objective.values())
        for i in range(len(v1)):
            if v1[i] > v2[i]:
                return 0  # 但凡有一个位置是 v1大于v2的 直接返回0,如果相等的话比较下一个目标值
        return 1


# TODO 如果目标函数是3个及以上呢，如果比较的支配方向不是越小越好呢？
def main():
    # 初始化/参数设置
    generations = 250  # 迭代次数
    popnum = 100  # 种群大小
    eta = 1  # 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1

    # poplength = 30  # 单个个体解向量的维数
    # bound_min = 0  # 定义域
    # bound_max = 1
    # objective_fun = ZDT1

    poplength = 3  # 单个个体解向量的维数
    bound_min = -5  # 定义域
    bound_max = 5
    objective_fun = KUR

    # 生成第一代种群
    P = []
    for i in range(popnum):
        P.append(Individual())
        P[i].solution = np.random.rand(poplength) * (bound_max - bound_min) + bound_min  # 随机生成个体可行解
        P[i].bound_process(bound_min, bound_max)  # 定义域越界处理
        P[i].calculate_objective(objective_fun)  # 计算目标函数值

    # 否 -> 非支配排序
    fast_non_dominated_sort(P)
    Q = make_new_pop(P, eta, bound_min, bound_max, objective_fun)

    P_t = P  # 当前这一届的父代种群
    Q_t = Q  # 当前这一届的子代种群

    for gen_cur in range(generations):
        R_t = P_t + Q_t  # combine parent and offspring population
        F = fast_non_dominated_sort(R_t)

        P_n = []  # 即为P_t+1,表示下一届的父代
        i = 1
        while len(P_n) + len(F[i]) < popnum:  # until the parent population is filled
            crowding_distance_assignment(F[i])  # calculate crowding-distance in F_i
            P_n = P_n + F[i]  # include ith non dominated front in the parent pop
            i = i + 1  # check the next front for inclusion
        F[i].sort(key=lambda x: x.distance)  # sort in descending order using <n，因为本身就在同一层，所以相当于直接比拥挤距离
        P_n = P_n + F[i][:popnum - len(P_n)]
        Q_n = make_new_pop(P_n, eta, bound_min, bound_max,
                           objective_fun)  # use selection,crossover and mutation to create a new population Q_n

        # 求得下一届的父代和子代成为当前届的父代和子代，，进入下一次迭代 《=》 t = t + 1
        P_t = P_n
        Q_t = Q_n

        # 绘图
        plt.clf()
        plt.title('current generation:' + str(gen_cur + 1))
        plot_P(P_t)
        plt.pause(0.1)

    plt.show()

    return 0


def fast_non_dominated_sort(P):
    """
    非支配排序
    :param P: 种群 P
    :return F: F=(F_1, F_2, ...) 将种群 P 分为了不同的层， 返回值类型是dict，键为层号，值为 List 类型，存放着该层的个体
    """
    F = defaultdict(list)

    for p in P:
        p.S = []
        p.n = 0
        for q in P:
            if p < q:  # if p dominate q
                p.S.append(q)  # Add q to the set of solutions dominated by p
            elif q < p:
                p.n += 1  # Increment the domination counter of p
        if p.n == 0:
            p.rank = 1
            F[1].append(p)

    i = 1
    while F[i]:
        Q = []
        for p in F[i]:
            for q in p.S:
                q.n = q.n - 1
                if q.n == 0:
                    q.rank = i + 1
                    Q.append(q)
        i = i + 1
        F[i] = Q

    return F


def crowding_distance_assignment(L):
    """ 传进来的参数应该是L = F(i)，类型是List"""
    l = len(L)  # number of solution in F

    for i in range(l):
        L[i].distance = 0  # initialize distance

    for m in L[0].objective.keys():
        L.sort(key=lambda x: x.objective[m])  # sort using each objective value
        L[0].distance = float('inf')
        L[l - 1].distance = float('inf')  # so that boundary points are always selected

        # 排序是由小到大的，所以最大值和最小值分别是 L[l-1] 和 L[0]
        f_max = L[l - 1].objective[m]
        f_min = L[0].objective[m]


        # 当某一个目标方向上的最大值和最小值相同时，此时会发生除零错，这里采用异常处理机制来解决
        try:
            for i in range(1, l - 1):  # for all other points
                L[i].distance = L[i].distance + (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)
        except Exception:
            print(str(m) + "目标方向上，最大值为" + str(f_max) + "最小值为" + str(f_min))


def binary_tournament(ind1, ind2):
    """
    二元锦标赛
    :param ind1:个体1号
    :param ind2: 个体2号
    :return:返回较优的个体
    """
    if ind1.rank != ind2.rank:  # 如果两个个体有支配关系，即在两个不同的rank中，选择rank小的
        return ind1 if ind1.rank < ind2.rank else ind2
    elif ind1.distance != ind2.distance:  # 如果两个个体rank相同，比较拥挤度距离，选择拥挤读距离大的
        return ind1 if ind1.distance > ind2.distance else ind2
    else:  # 如果rank和拥挤度都相同，返回任意一个都可以
        return ind1


# TODO 真想把 eta, bound_min, bound_max, objective_fun 设为全局变量
def make_new_pop(P, eta, bound_min, bound_max, objective_fun):
    """
    use select,crossover and mutation to create a new population Q
    :param P: 父代种群
    :param eta: 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1
    :param bound_min: 定义域下限
    :param bound_max: 定义域上限
    :param objective_fun: 目标函数
    :return Q : 子代种群
    """
    popnum = len(P)
    Q = []
    # binary tournament selection
    for i in range(int(popnum / 2)):
        # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent1
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent1 = binary_tournament(P[i], P[j])

        # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent2
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent2 = binary_tournament(P[i], P[j])

        while (parent1.solution == parent2.solution).all():  # 如果选择到的两个父代完全一样，则重选另一个
            i = random.randint(0, popnum - 1)
            j = random.randint(0, popnum - 1)
            parent2 = binary_tournament(P[i], P[j])

        # parent1 和 parent1 进行交叉，变异 产生 2 个子代
        Two_offspring = crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun)

        # 产生的子代进入子代种群
        Q.append(Two_offspring[0])
        Q.append(Two_offspring[1])
    return Q


def crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun):
    """
    交叉方式使用二进制交叉算子（SBX），变异方式采用多项式变异（PM）
    :param parent1: 父代1
    :param parent2: 父代2
    :param eta: 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1
    :param bound_min: 定义域下限
    :param bound_max: 定义域上限
    :param objective_fun: 目标函数
    :return: 2 个子代
    """
    poplength = len(parent1.solution)

    offspring1 = Individual()
    offspring2 = Individual()
    offspring1.solution = np.empty(poplength)
    offspring2.solution = np.empty(poplength)

    # 二进制交叉
    for i in range(poplength):
        rand = random.random()
        beta = (rand * 2) ** (1 / (eta + 1)) if rand < 0.5 else (1 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
        offspring1.solution[i] = 0.5 * ((1 + beta) * parent1.solution[i] + (1 - beta) * parent2.solution[i])
        offspring2.solution[i] = 0.5 * ((1 - beta) * parent1.solution[i] + (1 + beta) * parent2.solution[i])

    # 多项式变异
    # TODO 变异的时候只变异一个，不要两个都变，不然要么出现早熟现象，要么收敛速度巨慢 why？
    for i in range(poplength):
        mu = random.random()
        delta = 2 * mu ** (1 / (eta + 1)) if mu < 0.5 else (1 - (2 * (1 - mu)) ** (1 / (eta + 1)))
        offspring1.solution[i] = offspring1.solution[i] + delta

    # 定义域越界处理
    offspring1.bound_process(bound_min, bound_max)
    offspring2.bound_process(bound_min, bound_max)

    # 计算目标函数值
    offspring1.calculate_objective(objective_fun)
    offspring2.calculate_objective(objective_fun)

    return [offspring1, offspring2]


def ZDT1(x):
    """
    测试函数——ZDT1
    :parameter
    :param x: 为 m 维向量，表示个体的具体解向量
    :return f: 为两个目标方向上的函数值
    """
    poplength = len(x)
    f = defaultdict(float)

    g = 1 + 9 * sum(x[1:poplength]) / (poplength - 1)
    f[1] = x[0]
    f[2] = g * (1 - pow(x[0] / g, 0.5))

    return f


def ZDT2(x):
    poplength = len(x)
    f = defaultdict(float)

    g = 1 + 9 * sum(x[1:poplength]) / (poplength - 1)
    f[1] = x[0]
    f[2] = g * (1 - (x[0] / g) ** 2)

    return f


def KUR(x):
    f = defaultdict(float)
    poplength = len(x)

    f[1] = 0
    f[2] = 0
    for i in range(poplength - 1):
        f[1] = f[1] + (-10) * math.exp((-0.2) * (x[i] ** 2 + x[i+1] ** 2) ** 0.5)

    for i in range(poplength):
        f[2] = f[2] + abs(x[i]) ** 0.8 + 5 * math.sin(x[i] ** 3)

    return f

def plot_P(P):
    """
    假设目标就俩,给个种群绘图
    :param P:
    :return:
    """
    X = []
    Y = []
    for ind in P:
        X.append(ind.objective[1])
        Y.append(ind.objective[2])

    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.scatter(X, Y)


def show_some_ind(P):
    # 测试使用
    for i in P:
        print(i.solution)


if __name__ == '__main__':
    main()
