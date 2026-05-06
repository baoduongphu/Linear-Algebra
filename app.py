import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def initializeMatrix(n):
    print("--- KIỂM TRA ĐIỀU KIỆN CHÉO HÓA MA TRẬN n x n ---")
    print(f"Nhập các phần tử của ma trận {n}x{n} theo từng hàng (cách nhau bởi dấu cách):")
    matrix_data = []
    for i in range(n):
        while(True):
            row = list(map(float, input(f"Hàng {i+1}: ").split()))
            if len(row) != n:
                print(f"Lỗi: Hàng phải có đúng {n} phần tử.Yêu cầu nhập lại")
            else:
                matrix_data.append(row)
                break
    return np.array(matrix_data)

def diagonalize(A):
    eigenvals, eigenvecs = np.linalg.eig(A)

    print("\n" + "="*30)
    print("KẾT QUẢ TÍNH TOÁN:")
    print("Trị riêng (Eigenvalues):", eigenvals)
    print("Ma trận Vector riêng (P):\n", eigenvecs)
    print("="*30)
    
    rank_P = np.linalg.matrix_rank(eigenvecs)
    det_P = np.linalg.det(eigenvecs)
    is_diag = rank_P == len(A) and det_P != 0

    if is_diag:
        print("KẾT LUẬN: MA TRẬN CHÉO HÓA ĐƯỢC")
    else: 
        print("KẾT LUẬN: MA TRẬN KHÔNG CHÉO HÓA ĐƯỢC")
        
    return eigenvals, eigenvecs, is_diag
    
def input_parameters(n):
    """Hàm nhập điều kiện đầu x(0) và vector hàm cưỡng bức f(t) cho hệ bậc n"""
    print(f"\n--- NHẬP THÔNG SỐ HỆ VI PHÂN BẬC {n} ---")
    
    # 1. Nhập x(0)
    while True:
        try:
            print(f"1. Nhập vector x(0) ({n} phần tử, cách nhau bởi dấu cách):")
            x0 = np.array(list(map(float, input("   x(0) = ").split())))
            if len(x0) == n:
                break
            print(f"Lỗi: Bạn phải nhập đúng {n} phần tử.")
        except ValueError:
            print("Lỗi: Vui lòng chỉ nhập số.")
    
    # 2. Nhập từng thành phần của f(t)
    print(f"\n2. Nhập các thành phần của vector f(t):")
    print("   (Dùng 'np.sin', 'np.exp', 't' làm biến. Ví dụ: '10*np.sin(t)')")
    f_elements = []
    for i in range(n):
        element = input(f"   f{i+1}(t) = ")
        f_elements.append(element)
    
    # Chuyển list các chuỗi thành một chuỗi dạng list để eval
    f_str = f"[{",".join(f_elements)}]"
    return x0, f_str

def solve_non_homogeneous(A, n, vals, vecs, x0, f_str, t):
    P = vecs
    P_inv = np.linalg.inv(P)

    exp_At = P @ np.diag(np.exp(vals * t)) @ P_inv

    def integrand(s, idx):
        f_tau = np.array(eval(f_str, {"np": np, "t": s}))
        # e^{-A*tau} = P * diag(e^{-lambda*tau}) * P^-1
        exp_minus_Atau = P @ np.diag(np.exp(-vals * s)) @ P_inv
        v_tau = exp_minus_Atau @ f_tau
        return v_tau[idx].real

    # 3. Tính tích phân của v(tau) từ 0 đến t
    integral_vec = np.zeros(n)
    for i in range(n):
        res, _ = quad(integrand, 0, t, args=(i,))
        integral_vec[i] = res
        
    # 4. Nghiệm cuối cùng: x(t) = e^{At} * x0 + e^{At} * [Tích phân]
    # Có thể viết gọn thành: x(t) = e^{At} * (x0 + [Tích phân])
    x_total = exp_At @ (x0 + integral_vec)

    return x_total.real

def plot_system_solution(A, n, vals, vecs, x0, f_str, t_max):
    ts = np.linspace(0, t_max, 500)
    x_ts = np.zeros((len(ts), n))

    for i in range(len(ts)):
        t = ts[i]
        x_t = solve_non_homogeneous(A, n, vals, vecs, x0, f_str, t)
        x_ts[i, :] = x_t
    
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.plot(ts, x_ts[:, i], label=f'$x_{i+1}(t)$', linewidth=2)

    plt.title(f'Mô phỏng chuỗi thời gian hệ bậc {n}', fontsize=14)
    plt.xlabel('Thời gian (t)', fontsize=12)
    plt.ylabel('Giá trị biên độ', fontsize=12)
    plt.axhline(0, color='black', lw=1, linestyle='--')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    
    n = int(input("Nhập bậc n của ma trận vuông: "))
    
    matrix = initializeMatrix(n)
    vals, vecs, is_diagonalizable = diagonalize(matrix)

    if is_diagonalizable:
        x0, f_str = input_parameters(n)
        t_max = float(input("Nhập khoảng thời gian muốn quan sát (ví dụ 10s): "))
        plot_system_solution(matrix, n, vals, vecs, x0, f_str, t_max)
    else:
        print("Dừng chương trình vì ma trận không chéo hóa được.")
