from loguru import logger
from tqdm import tqdm
import numpy as np

from tools import get_memory

def log_base_n(n, x):
    return np.log(x) / np.log(n)

class Algorithm:
    def __init__(self, e_arr, b_arr, f, k, epsilon):
        self.e_arr = e_arr
        self.b_arr = b_arr
        self.f = f
        self.k = k
        self.epsilon = epsilon
        self.memory = 0

class Algorithm2(Algorithm):
    def __init__(self, e_arr, b_arr, f, k, epsilon):
        Algorithm.__init__(self, e_arr, b_arr, f, k, epsilon)

    def __generate_o(self, m):
        o_min = int(np.ceil(log_base_n(1 + self.epsilon, m)))
        o_max = int(np.floor(log_base_n(1 + self.epsilon, 2*self.k*m)))
        o_power = np.arange(o_min, o_max+1)
        o_base = np.full(len(o_power), 1 + self.epsilon)
        o_arr = list(set(np.ceil(np.power(o_base, o_power)).astype(int)))
        o_arr.sort()
        return o_arr

    def __generate_i(self, be):
        i_max = int(np.floor(log_base_n((1 - self.epsilon), 1/be)))
        i_min = 0 
        i_power = np.arange(i_min, i_max+1)
        i_base = np.full(len(i_power), 1 - self.epsilon)
        i_arr = set(np.ceil(np.power(i_base, i_power)*be).astype(int))
        i_arr = list(i_arr)
        i_arr.sort()
        return i_arr

    def __binary_search(self, x_arr, xe, i_arr, v):
        l = 0
        r = len(i_arr) - 1
        xv = x_arr[v]
        fx = self.f(xv)
        if self.f(xv + i_arr[0] * xe) - fx < i_arr[0] * v / (2 * self.k):
            return i_arr[0]
        while r > l:
            m = (l + r) // 2
            df = self.f(xv + i_arr[m] * xe) - fx
            threshold = i_arr[m] * v / (2 * self.k)
            if df >= threshold:
                l = m + 1
                continue
            r = m - 1
        return i_arr[l]

    @logger.catch
    def run(self):
        x_arr = dict()
        m = 0
        n = len(self.e_arr)
        with tqdm(total=n, leave=False, desc="Algorithm 2") as pbar:
            for e in self.e_arr:
                xe = np.full(n, 0)
                xe[e] = 1
                m = max(self.f(xe), m)
                o_arr = self.__generate_o(m)
                i_arr = self.__generate_i(self.b_arr[e])
                for v in o_arr:
                    if v not in x_arr.keys():
                        x_arr[v] = np.full(n, 0)
                    ke = self.__binary_search(x_arr, xe, i_arr, v)
                    knew = min(ke, self.k - np.sum(x_arr[v]))
                    if knew > 0:
                        x_arr[v] += knew*xe
                    else:
                        break
                pbar.update(1)
        x_list = [x_arr[key] for key in x_arr.keys()]
        fx_list = [self.f(x) for x in x_list]
        self.memory = get_memory()
        return x_list[np.argmax(fx_list)]

class Algorithm3(Algorithm):
    def __init__(self, e_arr, b_arr, f, k, epsilon):
        Algorithm.__init__(self, e_arr, b_arr, f, k, epsilon)

    def __generate_i(self, be):
        i_max = int(np.floor(log_base_n((1 - self.epsilon), 1/be)))
        i_min = 0 
        i_power = np.arange(i_min, i_max+1)
        i_base = np.full(len(i_power), 1 - self.epsilon)
        i_arr = set(np.ceil(np.power(i_base, i_power)*be).astype(int))
        return list(i_arr)

    def __find_ke(self, x, xe, i_arr):
        if len(i_arr) == 0:
            return 0
        if len(i_arr) == 1:
            return i_arr[0]

        fxe = lambda i: self.f(x + i_arr[i]*xe) - self.f(x + (i_arr[i]-1)*xe)

        first_condition = lambda t: fxe(t) < self.f(x + i_arr[t-1]*xe)/self.k
        second_condition = lambda t: np.all(
                [fxe(j) >= self.f(x + i_arr[j-1]*xe)/self.k 
                    for j in range(1, t+1)])
        ke_candidates = [i_arr[t]-1 for t in range(1, len(i_arr)) 
                            if first_condition(t) and second_condition(t)]
        try:
            return np.max(ke_candidates)
        except:
            return 0
        
    
    @logger.catch
    def run(self):
        n = len(self.e_arr)
        x = np.full(n, 1)
        with tqdm(total=n, leave=False, desc="Algorithm 3") as pbar:
            for e in self.e_arr:
                be = self.b_arr[e]
                xe = np.full(n, 0)
                xe[e] = 1
                i_arr = self.__generate_i(be)
                ke = self.__find_ke(x, xe, i_arr)
                x += ke * xe
                pbar.update(1)
        x_new = np.full(n, 0)
        has_at_least_one = False
        x_sum = 0
        for index in reversed(range(n)):
            if x[index] > 0 and not has_at_least_one:
                x_new[index] = x[index]
                has_at_least_one = True
                x_sum += x[index]
                continue
            if x_sum + x[index] > self.k:
                break
            x_new[index] = x[index]
            x_sum += x[index]
        self.memory = get_memory()
        return x_new

class Algorithm4(Algorithm):
    def __init__(self, e_arr, b_arr, f, k, epsilon):
        Algorithm.__init__(self, e_arr, b_arr, f, k, epsilon)

    def __binary_search(self, x, xe, i_arr, theta):
        try:
            l = 0
            r = len(i_arr) - 1
            fx = self.f(x)
            fi = lambda i: (self.f(x + i_arr[i]*xe) - fx)/i_arr[i]
            if fi(0) < theta:
                return i_arr[0]
            while r > l:
                m = (l + r) // 2
                df = (self.f(x + i_arr[m] * xe) - fx)/i_arr[m]
                if df >= theta:
                    l = m + 1
                else:
                    r = m - 1
            return i_arr[l] - 1
        except:
            return 0
    
    @logger.catch
    def run(self):
        algorithm3 = Algorithm3(self.e_arr, self.b_arr, self.f, self.k, self.epsilon)
        n = len(self.e_arr)
        x0 = algorithm3.run()
        gamma = self.f(x0)
        theta = (4 - 3*self.epsilon) * gamma /((1 - 3*self.epsilon) * self.k)
        x = np.full(n, 0)
        xe_dict = {}
        exit_threshold = (1 - self.epsilon) * gamma / (4 * self.k) 
        with tqdm(total=n, leave=False) as pbar:
            while theta >= exit_threshold:
                pbar.reset()
                desc = f'Algorithm 4 [{round(theta,2)} >= {round(exit_threshold,2)}]'
                pbar.set_description(desc)
                pbar.refresh()
                for e in self.e_arr:
                    if e not in xe_dict:
                        xe_dict[e] = np.full(n, 0)
                        xe_dict[e][e] = 1
                    be = self.b_arr[e]
                    i_arr = np.arange(1, be+1) 
                    ke = self.__binary_search(x, xe_dict[e], i_arr, theta)
                    k_new = min(ke, self.k - np.sum(x))
                    if k_new != 0:
                        x += xe_dict[e] * k_new
                    else:
                        break
                    pbar.update(1)
                theta = (1 - self.epsilon) * theta
                if theta == 0:
                    break
        self.memory = get_memory()
        return x

class ThresholdGreedy(Algorithm):
    def __init__(self, e_arr, b_arr, f, k, epsilon):
        Algorithm.__init__(self, e_arr, b_arr, f, k, epsilon)

    def __binary_search(self, x, e, tau):
        l = 1
        r = min(self.b_arr[e] - x[e], self.k - np.sum(x))
        xe = np.zeros(len(x))
        xe[e] = 1
        fx = self.f(x)
        if self.f(x + r*xe) - fx >= tau:
            return r
        if self.f(x + xe) - fx < tau:
            return 0
        while r > l + 1:
            m = (l + r) // 2
            if self.f(x + m*xe) - fx >= tau:
                l = m
                continue
            r = m
        return l

    @logger.catch
    def run(self):
        n = len(self.e_arr)

        def create_xe(e):
            xe = np.zeros(n)
            xe[e] = 1
            return xe

        x = np.zeros(n)
        ls_xe = [create_xe(e) for e in self.e_arr]
        d = max([self.f(xe) for xe in ls_xe])
        tau = d
        threshold = self.epsilon / self.k * d
        with tqdm(total = n, leave=False) as pbar:
            while tau >= threshold:
                pbar.reset()
                desc = f'ThGreedy [{round(tau, 2)} >= {round(threshold, 2)}]'
                pbar.set_description(desc)
                pbar.refresh()
                for e in self.e_arr:
                    l = self.__binary_search(x, e, tau)
                    x += l * ls_xe[e]
                    if np.sum(x) == self.k:
                        self.memory = get_memory()
                        return x
                    pbar.update(1)
                tau *= (1 - self.epsilon)
        self.memory = get_memory()
        return x

class SieveStreaming(Algorithm):
    def __init__(self, e_arr, b_arr, f, k, epsilon):
        Algorithm.__init__(self, e_arr, b_arr, f, k, epsilon)

    def __binary_search(self, x, xe, e, tau):
        l = 1
        r = min(self.b_arr[e] - x[e], self.k - np.sum(x))
        fx = self.f(x)
        if self.f(x + r*xe) - fx >= tau:
            return r
        if self.f(x + xe) - fx < tau:
            return 0
        while r > l + 1:
            m = (l + r) // 2
            if self.f(x + m*xe) - fx >= tau:
                l = m
                continue
            r = m
        return l

    def __generate_h(self, alpha, theta):
        gamma = 1 + self.epsilon
        h_min = int(np.ceil(log_base_n(gamma, theta / gamma)))
        h_max = int(np.floor(log_base_n(gamma, 2 * self.k * alpha)))
        h_power = np.arange(h_min, h_max + 1)
        h_base = np.full(len(h_power), gamma)
        h_arr = set(np.ceil(np.power(h_base, h_power)).astype(int))
        return h_arr
    
    @logger.catch
    def run(self):
        xv_dict = dict()
        alpha = 0
        theta = 0
        eta = 0
        n = len(self.e_arr)
        with tqdm(total=n, leave=False, desc='Sieve Streaming') as pbar:
            for e in self.e_arr:
                xe = np.zeros(n)
                xe[e] = 1
                alpha = max(alpha, self.f(xe))
                theta = max(alpha, eta)
                h = self.__generate_h(alpha, theta)
                v_previous = set(xv_dict.keys())
                # remove v that is outside of h
                for v in v_previous.difference(h):
                    xv_dict.pop(v)
                for v in h:
                    if v not in xv_dict:
                        xv_dict[v] = np.zeros(n)
                    xE = sum(xv_dict[v])
                    if xE < self.k:
                        tau = (v/2 - self.f(xv_dict[v])) / (self.k - xE)
                        l = self.__binary_search(xv_dict[v], xe, e, tau)
                        xv_dict[v] += l*xe
                fxv = [self.f(xv_dict[v]) for v in xv_dict.keys()]
                fxv.append(eta)
                eta = max(fxv)
                pbar.update(1)
        x_list = [xv_dict[key] for key in xv_dict.keys()]
        fx_list = [self.f(x) for x in x_list]
        self.memory = get_memory()
        return x_list[np.argmax(fx_list)]

class SomaCardinality(Algorithm):
    def __init__(self, e_arr, b_arr, f, k, epsilon):
        Algorithm.__init__(self, e_arr, b_arr, f, k, epsilon)

    def __binary_search(self, x, e, tau):
        l = 1
        r = min(self.b_arr[e] - x[e], self.k - np.sum(x))
        xe = np.zeros(len(x))
        xe[e] = 1
        fx = self.f(x)
        if self.f(x + r*xe) - fx >= r * tau:
            return r
        if self.f(x + xe) - fx < l * tau:
            return 0
        while r > l + 1:
            m = (l + r) // 2
            if self.f(x + m*xe) - fx >= m * tau:
                l = m
                continue
            r = m
        return l

    @logger.catch
    def run(self):
        n = len(self.e_arr)

        def create_xe(e):
            xe = np.zeros(n)
            xe[e] = 1
            return xe

        x = np.zeros(n)
        ls_xe = [create_xe(e) for e in self.e_arr]
        d = max([self.f(xe) for xe in ls_xe])
        tau = d
        threshold = self.epsilon / self.k * d
        with tqdm(total = n, leave=False) as pbar:
            while tau >= threshold:
                pbar.reset()
                desc = f'Soma Cardinality [{round(tau, 2)} >= {round(threshold, 2)}]'
                pbar.set_description(desc)
                pbar.refresh()
                for e in self.e_arr:
                    l = self.__binary_search(x, e, tau)
                    x += l * ls_xe[e]
                    pbar.update(1)
                tau *= (1 - self.epsilon)
        self.memory = get_memory()
        return x
