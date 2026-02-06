"""
橫場 Ising 模型 (Transverse Field Ising Model) 的 SSE QMC 模擬程式。
基於 A. W. Sandvik (arXiv:cond-mat/0303597v2) 的演算法。

H = Σ_{i,j} J_{ij} σz_i σz_j - h Σ_i σx_i

算符定義：
- H_{0,0} = 1 (identity)
- H_{i,0} = h * σx_i (spin flip)
- H_{i,i} = h (constant)
- H_{i,j} = |J_{ij}| - J_{ij}*σz_i*σz_j (Ising bond)

程式碼編碼 (op):
- (0, 0): identity
- (i, 0): spin-flip
- (i, i): constant
- (i, j): Ising bond
"""
import numpy as np

class SSE_TFIM:
    """
    1D TFIM SSE 模擬類別。
    H = -J Σ σz_i σz_{i+1} - h Σ σx_i
    採用週期性邊界條件。
    """

    def __init__(self, N, beta, J=1.0, h=1.0, L_init=10):
        """
        初始化參數：
        N: 格點數
        beta: 倒數溫度
        J: Ising 耦合常數 (我們令 H = -J*σz*σz，正值為鐵磁)
        h: 橫場強度
        L_init: 算符列表初始長度
        """
        self.N = N
        self.beta = beta
        self.J = abs(J)  # 取絕對值，符號在算符定義中處理
        self.h = h
        
        # 算符列表編碼：
        # type: 0=id, 1=flip, 2=const, 3=bond
        # site1, site2: 涉及的格點
        self.L = L_init
        self.op_type = np.zeros(L_init, dtype=np.int8)
        self.op_site1 = np.zeros(L_init, dtype=np.int16)
        self.op_site2 = np.zeros(L_init, dtype=np.int16)
        
        # 非單位算符的數量
        self.n = 0
        
        # 自旋組態 (0 或 1)
        self.spins = np.random.randint(0, 2, size=N)
        
        # 對角更新的權重總和 W_diag = N*h + N*2|J|
        self.W_diag = N * self.h + 2 * N * self.J

    def _expand(self, new_L):
        """擴充算符列表容量"""
        if new_L <= self.L:
            return
        new_type = np.zeros(new_L, dtype=np.int8)
        new_site1 = np.zeros(new_L, dtype=np.int16)
        new_site2 = np.full(new_L, -1, dtype=np.int16)
        
        new_type[:self.L] = self.op_type
        new_site1[:self.L] = self.op_site1
        new_site2[:self.L] = self.op_site2
        
        self.op_type = new_type
        self.op_site1 = new_site1
        self.op_site2 = new_site2
        self.L = new_L

    def diagonal_update(self):
        """
        對角更新：插入或移除對角算符 (Constant 或 Ising bond)。
        根據 Sandvik 論文公式 (14)-(15)。
        """
        spins = self.spins.copy()
        n = self.n
        N = self.N
        L = self.L
        beta = self.beta
        h = self.h
        
        for p in range(L):
            op_t = self.op_type[p]
            
            if op_t == 0:
                # Identity: 嘗試插入對角算符
                prob_insert = beta * self.W_diag / (L - n)
                
                if np.random.random() < prob_insert:
                    # 選擇插入類型：Constant 或 Bond
                    r = np.random.random() * self.W_diag
                    
                    if r < N * h:
                        # 插入 Constant (H_{i,i})
                        site = int(r / h) % N
                        self.op_type[p] = 2
                        self.op_site1[p] = site
                        self.op_site2[p] = -1
                        n += 1
                    else:
                        # 嘗試插入 Ising bond (H_{i,j})
                        bond = np.random.randint(0, N)
                        i = bond
                        j = (bond + 1) % N
                        
                        # 自旋平行 (鐵磁條件) 才可插入
                        if spins[i] == spins[j]:
                            self.op_type[p] = 3
                            self.op_site1[p] = min(i, j)
                            self.op_site2[p] = max(i, j)
                            n += 1
                            
            elif op_t == 2:
                # Constant: 嘗試移除
                prob_remove = (L - n + 1) / (beta * self.W_diag)
                if np.random.random() < prob_remove:
                    self.op_type[p] = 0
                    self.op_site1[p] = 0
                    self.op_site2[p] = -1
                    n -= 1
                    
            elif op_t == 3:
                # Ising bond: 嘗試移除
                prob_remove = (L - n + 1) / (beta * self.W_diag)
                if np.random.random() < prob_remove:
                    self.op_type[p] = 0
                    self.op_site1[p] = 0
                    self.op_site2[p] = -1
                    n -= 1
                    
            elif op_t == 1:
                # Spin-flip: 傳遞自旋狀態
                site = self.op_site1[p]
                spins[site] = 1 - spins[site]
        
        self.n = n
        
        # 若算符佔比過高則擴充列表
        if n > 0.7 * L:
            self._expand(int(1.5 * L) + 10)

    def off_diagonal_update(self):
        """
        非對角更新：交換「成對的 Constant」與「成對的 Spin-flip」。
        對應論文公式 (13)。
        """
        N = self.N
        L = self.L
        
        for site in range(N):
            site_ops = []
            constrained = []
            
            for p in range(L):
                if self.op_type[p] == 2 and self.op_site1[p] == site:
                    site_ops.append((p, 2))
                elif self.op_type[p] == 1 and self.op_site1[p] == site:
                    site_ops.append((p, 1))
                elif self.op_type[p] == 3:
                    if self.op_site1[p] == site or self.op_site2[p] == site:
                        constrained.append(p)
            
            if len(site_ops) < 2:
                continue
            
            # 隨機選兩算符嘗試交換
            for _ in range(len(site_ops) // 2):
                idx1, idx2 = np.random.choice(len(site_ops), 2, replace=False)
                p1, t1 = site_ops[idx1]
                p2, t2 = site_ops[idx2]
                
                if p1 > p2:
                    p1, p2 = p2, p1
                    t1, t2 = t2, t1
                
                # 檢查是否有 Ising bond 阻擋
                blocked = any(p1 < c < p2 for c in constrained)
                blocked = blocked or any(c > p2 or c < p1 for c in constrained if p2 < p1)
                
                if not blocked and t1 == t2:
                    new_type = 1 if t1 == 2 else 2
                    self.op_type[p1] = new_type
                    self.op_type[p2] = new_type

    def classical_cluster_update(self):
        """
        古典 Cluster 更新 (Swendsen-Wang)：
        根據 Ising bond 連結自旋，並以 1/2 機率翻轉各 Cluster。
        """
        N = self.N
        parent = list(range(N))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 根據 Ising bond 連結
        for p in range(self.L):
            if self.op_type[p] == 3:
                union(self.op_site1[p], self.op_site2[p])
        
        # 決定 Cluster 翻轉
        cluster_flip = {}
        for i in range(N):
            root = find(i)
            if root not in cluster_flip:
                cluster_flip[root] = (np.random.random() < 0.5)
        
        # 執行翻轉
        for i in range(N):
            if cluster_flip[find(i)]:
                self.spins[i] = 1 - self.spins[i]

    def quantum_cluster_update(self):
        """
        量子 Cluster 更新：
        這裡簡化為僅執行 Off-diagonal update (處理 spin-flip) 與 Classical cluster update。
        """
        N = self.N
        n_ops = self.n
        
        if n_ops == 0:
            for i in range(N):
                if np.random.random() < 0.5:
                    self.spins[i] = 1 - self.spins[i]
            return

        self.off_diagonal_update()
        self.classical_cluster_update()

    def mc_step(self):
        """執行一次完整的 Monte Carlo 步 (對角 + Cluster 更新)"""
        self.diagonal_update()
        self.quantum_cluster_update()

    def measure_energy(self):
        """
        計算 Ising 能量 (H_Ising = -J Σ σz σz)。
        """
        E_ising = 0.0
        for i in range(self.N):
            j = (i + 1) % self.N
            sz_i = 2 * self.spins[i] - 1
            sz_j = 2 * self.spins[j] - 1
            E_ising -= self.J * sz_i * sz_j
        return E_ising

    def measure_energy_sse(self):
        """
        使用 SSE 估算測量能量：E = -<n>/beta + const
        """
        n_bond = np.sum(self.op_type == 3)
        n_const = np.sum(self.op_type == 2)
        n_flip = np.sum(self.op_type == 1)
        n_total = n_bond + n_const + n_flip
        
        const = self.N * self.h + self.N * self.J
        return -n_total / self.beta + const

    def measure_mz(self):
        """測量磁化強度 |<Mz>|"""
        sz = 2 * self.spins - 1
        return abs(np.mean(sz))

    def run(self, n_therm=1000, n_measure=2000, n_skip=5):
        """執行模擬：熱平衡 + 測量"""
        # 熱平衡
        for _ in range(n_therm):
            self.mc_step()
        
        Es_ising = []
        Es_sse = []
        Ms = []
        
        # 測量
        for _ in range(n_measure):
            for _ in range(n_skip):
                self.mc_step()
            Es_ising.append(self.measure_energy())
            Es_sse.append(self.measure_energy_sse())
            Ms.append(self.measure_mz())
            
        return {
            'energy_ising': np.mean(Es_ising),
            'energy_ising_err': np.std(Es_ising) / np.sqrt(len(Es_ising)),
            'energy_sse': np.mean(Es_sse),
            'energy_sse_err': np.std(Es_sse) / np.sqrt(len(Es_sse)),
            'magnetization': np.mean(Ms),
            'magnetization_err': np.std(Ms) / np.sqrt(len(Ms)),
            'n_ops': self.n,
            'L': self.L
        }

if __name__ == "__main__":
    np.random.seed(42)
    print("Testing SSE for TFIM")
    print("=" * 50)
    
    # Test h=0 (pure Ising)
    print("\nh=0 (pure Ising):")
    sse = SSE_TFIM(N=4, beta=2.0, J=1.0, h=0.001, L_init=20)
    r = sse.run(n_therm=1000, n_measure=2000)
    print(f"  E_Ising = {r['energy_ising']:.4f} ± {r['energy_ising_err']:.4f}")
    print(f"  E_SSE = {r['energy_sse']:.4f}")
    print(f"  |mz| = {r['magnetization']:.4f}")
    print(f"  n={r['n_ops']}, L={r['L']}")
    
    # Test h=1
    print("\nh=1:")
    sse = SSE_TFIM(N=4, beta=2.0, J=1.0, h=1.0, L_init=20)
    r = sse.run(n_therm=1000, n_measure=2000)
    print(f"  E_Ising = {r['energy_ising']:.4f}")
    print(f"  E_SSE = {r['energy_sse']:.4f}")
    print(f"  |mz| = {r['magnetization']:.4f}")