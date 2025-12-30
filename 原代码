using DifferentialEquations
using Flux
using DiffEqFlux
using Interpolations
using Optimization
using OptimizationOptimisers
using OptimizationPolyalgorithms
using ForwardDiff
using Statistics
using LinearAlgebra
using Printf
using Plots
using CSV
using DataFrames
using Dates
using Measures
using InlineStrings 

# --- 0. 准备工作 ---
const OUTPUT_DIR = "结果输出"
if !isdir(OUTPUT_DIR); mkdir(OUTPUT_DIR); end
const F = Float64
const DEPTH_THRESHOLD = 15.0 # *** 土层分界阈值m***

# ========================================================
# 辅助函数：强制将数据转换为 Float64
# ========================================================
function force_parse_float(x)
    if x isa Number
        return Float64(x)
    elseif x === missing
        return 0.0 
    else
        s = string(x)
        v = tryparse(Float64, s)
        return isnothing(v) ? 0.0 : v
    end
end

# ========================================================
# [物理参数映射] - 保持之前的设定
# ========================================================
function get_phys_params_mapped(p)
    # --- 1. Topsoil 参数 ---
    log_alpha_top = -4.5 + 3.5 * sigmoid(p[1])   
    n_top         = 1.05 + 0.55 * sigmoid(p[2])   
    log_Ks_top    = log(0.5) + (log(100.0) - log(0.5)) * sigmoid(p[3]) 
    theta_s_top   = 0.30 + 0.25 * sigmoid(p[7]) 
    
    # --- 2. Subsoil 参数 ---
    log_alpha_sub = log_alpha_top - (0.1 + 1.5 * sigmoid(p[4])) 
    n_sub         = 1.05 + (n_top - 1.05 ) * sigmoid(p[5]) 
    log_Ks_sub    = log_Ks_top - (0.5 + 3.0 * sigmoid(p[6])) 
    theta_s_sub   = 0.20 + (theta_s_top - 0.20) * sigmoid(p[9]) 
    
    # --- 3. 公共参数 ---
    theta_r_val   = 0.001 + 0.08 * sigmoid(p[8]) 
    log_I_max_val = log(0.001) + (log(10.0) - log(0.001)) * sigmoid(p[10]) 
    alpha_et      = 0.05 + 0.45 * sigmoid(p[11]) 
    root_decay    = 20.0 + 80.0 * sigmoid(p[12]) 

    return [log_alpha_top, n_top, log_Ks_top, 
            log_alpha_sub, n_sub, log_Ks_sub,
            theta_s_top, theta_r_val, theta_s_sub, 
            log_I_max_val, alpha_et, root_decay]
end

# --- 1. 物理模型 (VGM) ---
function vgm_robust_fixed(h, VGM_params)
    α = exp(VGM_params[1])
    n = VGM_params[2]
    Ks = exp(VGM_params[3])
    θs, θr = VGM_params[4], VGM_params[5]
    
    T = eltype(h)
    epsilon = T(1e-5)
    
    if h < -epsilon
        m = 1.0 - 1.0/n
        abs_h = abs(h)
        denom = (1.0 + (α * abs_h)^n)
        Se = denom^(-m)
        theta = θr + (θs - θr) * Se
        num = (θs - θr) * m * n * α^n * (abs_h^(n-1))
        C = num / (denom^(m+1))
        K_rel = sqrt(Se) * (1.0 - (1.0 - Se^(1/m))^m)^2
        return theta, max(C, T(1e-4)), max(Ks * K_rel, T(1e-8))
    else
        return θs, T(1e-3), Ks
    end
end

# 反推水头函数
function inverse_vgm_robust(theta_val, VGM_params)
    α = exp(VGM_params[1])
    n = VGM_params[2]
    θs, θr = VGM_params[4], VGM_params[5]
    m = 1.0 - 1.0/n
    th_safe = max(θr + 1e-4, min(θs - 1e-4, theta_val))
    Se = (th_safe - θr) / (θs - θr)
    term = Se^(-1.0/m) - 1.0
    return term < 0 ? 0.0 : -((term^(1.0/n)) / α)
end

# *** 小时级蒸散发模型 (含昼夜节律，已移除1400步后的夏季增强) ***
function get_Ep_seasonal(t)
    hour_of_day = t % 24
    diurnal = max(0.0, sin(pi * (hour_of_day - 6) / 12))
    season_factor = 0.5 # 常数因子
    return diurnal * season_factor
end

# --- 2. 神经网络 ---
nn_model = Flux.Chain(
    Flux.Dense(2, 10, tanh),
    Flux.Dense(10, 1, sigmoid)
)
nn_model = Flux.f64(nn_model)
p_nn_init, re_nn = Flux.destructure(nn_model)
p_nn_init[end] = 0.0 

# --- 3. ODE 系统 ---
struct RichardsParams
    z_nodes::Vector{F}
    rain_itp::AbstractInterpolation
    obs_indices::Vector{Int}
end

function richards_ode_improved!(dhdt, h, p, t, const_params::RichardsParams)
    p_phy = get_phys_params_mapped(p[1:12]) 
    p_top = [p_phy[1], p_phy[2], p_phy[3], p_phy[7], p_phy[8]]
    p_sub = [p_phy[4], p_phy[5], p_phy[6], p_phy[9], p_phy[8]]
    
    I_max = exp(p_phy[10])
    alpha_et = p_phy[11]
    root_decay = p_phy[12]
    z = const_params.z_nodes 
    dz = diff(z)
    N = length(h)

    # 1. 计算 C 和 K
    K = zeros(eltype(h), N)
    C = zeros(eltype(h), N)
    for i in 1:N
        w = 1.0 - sigmoid((z[i] - DEPTH_THRESHOLD) / 5.0)
        curr_p = [
            w*p_top[1] + (1-w)*p_sub[1],
            w*p_top[2] + (1-w)*p_sub[2],
            w*p_top[3] + (1-w)*p_sub[3],
            w*p_top[4] + (1-w)*p_sub[4],
            p_top[5]
        ]
        _, C[i], K[i] = vgm_robust_fixed(max(-1e8, min(100.0, h[i])), curr_p)
    end
    
    # 2. 计算通量 q
    q = Vector{eltype(h)}(undef, N-1)
    for i in 1:N-1
        K_mid = sqrt(K[i] * K[i+1])
        d_h = h[i+1] - h[i]
        q[i] = -K_mid * (d_h / dz[i] - 1.0)
    end
    
    # 3. 蒸散发逻辑 (NN + Climate)
    h_surf = h[1]
    input_h = tanh(log10(max(1.0, abs(h_surf))) / 5.0)
    
    # *** 关键修正：移除 NN 输入中的时间开关 ***
    # input_season = sigmoid((t - 1400.0) / 100.0)  <-- 移除此行
    input_season = 0.0 # 设置为常数，确保 NN 只响应水分变化，不响应时间
    
    beta = re_nn(p[13:end])([input_h, input_season])[1]
    
    E_pot = get_Ep_seasonal(t)
    E_s_act = beta * (alpha_et * E_pot)         
    T_pot_scaled = beta * ((1.0 - alpha_et) * E_pot)   

    # 4. 优先流逻辑
    Q_rain = const_params.rain_itp(t)
    Q_in_matrix_pot = min(Q_rain, I_max) 
    Q_bypass = max(0.0, Q_rain - Q_in_matrix_pot)
    
    idx_bypass_start = 6 
    idx_bypass_end = 10  
    lambda_bypass = 150.0 
    
    bypass_weights_sum = 0.0
    for j in idx_bypass_start:idx_bypass_end
        depth_rel = z[j] - z[idx_bypass_start]
        bypass_weights_sum += exp(-depth_rel / lambda_bypass)
    end

    # 5. 根系权重
    root_weights = zeros(eltype(h), N)
    total_root_weight = 0.0
    for i in 1:N
        base_root = 0.05 
        root_weights[i] = exp(-z[i] / root_decay) + base_root
        total_root_weight += root_weights[i] * (i==1 ? dz[1] : (i==N ? dz[N-1] : 0.5*(dz[i-1]+dz[i])))
    end

    # 6. 计算 dhdt
    for i in 1:N
        local_stress = sigmoid((h[i] + 15000.0) / 2000.0)
        sink_rate = T_pot_scaled * (root_weights[i] / total_root_weight) * local_stress
        
        if i == 1
            vol = 0.5 * dz[1]
            flux_in = (h[1] >= -10.0) ? min(Q_in_matrix_pot, K[1]) : Q_in_matrix_pot
            surf_stress = sigmoid((h[1] + 50000.0) / 5000.0)
            flux_out = q[1] + (E_s_act * 0.5 * surf_stress) 
        elseif i == N
            vol = 0.5 * dz[N-1]
            flux_in = q[N-1]
            flux_out = K[N] 
        else
            vol = 0.5 * (dz[i-1] + dz[i])
            flux_in = q[i-1]
            flux_out = q[i]
        end
        
        net_flux = (flux_in - flux_out) / vol - sink_rate
        
        if i == 2
             sub_stress = sigmoid((h[2] + 50000.0) / 5000.0)
             net_flux -= (E_s_act * 0.5 * sub_stress) / vol
        end
        
        if i >= idx_bypass_start && i <= idx_bypass_end && Q_bypass > 1e-5
             depth_rel = z[i] - z[idx_bypass_start]
             w_i = exp(-depth_rel / lambda_bypass)
             q_inject = Q_bypass * (w_i / bypass_weights_sum)
             net_flux += q_inject / vol
        end

        dhdt[i] = net_flux / C[i]
    end
end

# --- 4. 数据加载 (使用图片标题的行号 19658 to 21840) ---
println("=== 4. 数据加载与初始化 ===")
CSV_PATH = "g:/任鹏辉/孔老师任务/数据/USCRN-Filled_hourly_2024_sp54_Apr-Jun-1.csv" 
df = CSV.read(CSV_PATH, DataFrame; missingstring=["", "NA", "NaN", "null", "-9999"])

# [数据截取]：使用图片标题所示行数
const START_ROW = 24026
const END_ROW = 26208
const N_SLICED_STEPS = END_ROW - START_ROW + 1 

if nrow(df) >= END_ROW
    println(">>> 已截取第 $START_ROW 行至第 $END_ROW 行 (共 $N_SLICED_STEPS 个数据点) 进行模拟。")
    df = df[START_ROW:END_ROW, :]
else
    error(">>> 错误：数据行数 ($(nrow(df))) 少于请求的结束行 ($END_ROW)。无法截取数据。")
end

raw_rain_mm = force_parse_float.(df[:, 15])
raw_rain_cm = Float64.(raw_rain_mm ./ 10.0)

N_total_steps = nrow(df)
t_data = Float64.(0:(N_total_steps-1)) 
rain_itp = LinearInterpolation(t_data, raw_rain_cm, extrapolation_bc=Flat())
rain_indices = findall(x -> x > 0.0, raw_rain_cm)
rain_times = t_data[rain_indices]
tstops_points = sort(unique(vcat(rain_times, rain_times .+ 0.5))) 

sm_cols = Matrix(df[:, 31:35])
raw_sm_vec = force_parse_float.(sm_cols)
raw_sm = raw_sm_vec' |> collect 

if mean(raw_sm) > 1.0; raw_sm .= raw_sm ./ 100.0; end

obs_depths = [5.0, 10.0, 20.0, 50.0, 100.0]
# *** 精简后的网格节点 ***
z_nodes = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 50.0, 75.0, 100.0]
obs_indices = [findfirst(x -> x==d, z_nodes) for d in obs_depths]

# *** 动态初始水头计算辅助函数 (保持不变) ***
function get_initial_h_vector(p_mapped, obs_t0)
    T = eltype(p_mapped) 
    itp_init = LinearInterpolation(obs_depths, obs_t0, extrapolation_bc=Line())
    
    params_top = [p_mapped[1], p_mapped[2], p_mapped[3], p_mapped[7], p_mapped[8]]
    params_sub = [p_mapped[4], p_mapped[5], p_mapped[6], p_mapped[9], p_mapped[8]]
    
    h0 = zeros(T, length(z_nodes)) 
    
    for i in 1:length(z_nodes)
        z_val = z_nodes[i]
        par = (z_val <= DEPTH_THRESHOLD) ? params_top : params_sub
        h0[i] = inverse_vgm_robust(itp_init(z_val), par)
    end
    return h0
end

# *** 修正 p_phys_init 列表：针对新时段的极干燥环境 ***
p_phys_init = [
     0.8,  3.0,  1.5,  # 1-3: Topsoil α(↑增大透气性), n(↑最大化陡度), Ks
    -3.0,  3.0, -3.0,  # 4-6: Subsoil α_diff(↓使深层α接近表层), n_diff(↑最大化深层n), Ks_diff(↓保持高导水)
    -1.5, -2.5, -2.0,  # 7-9: θs_top, θr, θs_sub(↓大幅降低深层饱和含水量)
     0.5, -2.0,  3.0   # 10-12: Imax, αet(↓增加蒸腾比重), RootDecay(↑根系更深)
]
p_all_init = [p_phys_init; p_nn_init] 
h_init_initial = get_initial_h_vector(get_phys_params_mapped(p_phys_init), raw_sm[:, 1])

const_params = RichardsParams(z_nodes, rain_itp, obs_indices)
prob = ODEProblem((du, u, p, t) -> richards_ode_improved!(du, u, p, t, const_params), 
                  h_init_initial, (t_data[1], t_data[end]), p_all_init)

# --- 5. 训练流程 ---
println("=== 5. 开始模型训练 ===")

# ******************************************************
# *** 打印初始物理参数的原始潜变量和映射值 ***
# ******************************************************
println("\n>>> 初始参数设置:")
@printf("  原始潜变量 (p[1:12]): %s\n", join([@sprintf("%.2f", p) for p in p_phys_init], ", "))

p_init_mapped = get_phys_params_mapped(p_phys_init)
@printf("  初始映射值:\n")
@printf("    Top: α_log:%.2f, n:%.2f, Ks_log:%.2f, θs:%.3f | ", p_init_mapped[1], p_init_mapped[2], p_init_mapped[3], p_init_mapped[7])
@printf("Sub: α_log:%.2f, n:%.2f, Ks_log:%.2f, θs:%.3f |\n", p_init_mapped[4], p_init_mapped[5], p_init_mapped[6], p_init_mapped[9]) 
@printf("    Comm: θr:%.3f, I_max_log:%.3f, α_et:%.3f, RootZ:%.1f\n", 
         p_init_mapped[8], p_init_mapped[10], p_init_mapped[11], p_init_mapped[12]) 
# ******************************************************


function loss_fn_physics(p, target_sm, steps)
    t_end = t_data[steps]
    p_mapped = get_phys_params_mapped(p[1:12])
    T = eltype(p)
    
    # *** 关键：动态计算初始水头并更新 ODEProblem ***
    u0_dynamic = get_initial_h_vector(p_mapped, target_sm[:, 1])
    _prob = remake(prob, u0=u0_dynamic, tspan=(T(0.0), T(t_end)), p=p)

    _sol = solve(_prob, Rodas5(autodiff=false), saveat=t_data[1:steps],
                 tstops=filter(x->x<=t_end, tstops_points),
                 abstol=1e-2, reltol=1e-2, maxiters=100000)
    
    if _sol.retcode != :Success; return T(1e7); end
    
    params_top = [p_mapped[1], p_mapped[2], p_mapped[3], p_mapped[7], p_mapped[8]]
    params_sub = [p_mapped[4], p_mapped[5], p_mapped[6], p_mapped[9], p_mapped[8]] 

    loss = 0.0
    for (k, grid_idx) in enumerate(obs_indices)
        sim_h = _sol[grid_idx, :]
        depth_val = z_nodes[grid_idx]
        par = (depth_val <= DEPTH_THRESHOLD) ? params_top : params_sub
        sim_theta = [vgm_robust_fixed(h, par)[1] for h in sim_h]
        loss += sum(abs2, sim_theta .- target_sm[k, 1:steps]) * 10.0
    end
    return loss
end

# ******************************************************
# *** callback 函数 ***
# ******************************************************
callback = function (state, l)
    u_curr = (state isa AbstractArray) ? state : state.u
    p_dec = get_phys_params_mapped(u_curr[1:12]) 
    
    @printf("\nLoss: %.4f |\n", l)
    @printf("  Raw p[1:12]: %s\n", join([@sprintf("%.2f", p) for p in u_curr[1:12]], ", "))
    
    @printf("  Mapped: Top: α_log:%.2f, n:%.2f, Ks_log:%.2f, θs:%.3f | ", p_dec[1], p_dec[2], p_dec[3], p_dec[7])
    @printf("Sub: α_log:%.2f, n:%.2f, Ks_log:%.2f, θs:%.3f |\n", p_dec[4], p_dec[5], p_dec[6], p_dec[9]) 
    @printf("          Comm: θr:%.3f, I_max_log:%.3f, α_et:%.3f, RootZ:%.1f\n", 
             p_dec[8], p_dec[10], p_dec[11], p_dec[12]) 
    
    return false
end
# ******************************************************

println("\n>>> 阶段 1: 热身...")
steps_warmup = min(300, N_total_steps)
optf_1 = Optimization.OptimizationFunction((x,p)->loss_fn_physics(x, raw_sm, steps_warmup), Optimization.AutoForwardDiff())
optprob_1 = Optimization.OptimizationProblem(optf_1, p_all_init)
res_1 = Optimization.solve(optprob_1, OptimizationOptimisers.Adam(0.05), maxiters=150, callback=callback) 

println("\n>>> 2. 全局训练...")
steps_full = min(1700, N_total_steps)
optf_2 = Optimization.OptimizationFunction((x,p)->loss_fn_physics(x, raw_sm, steps_full), Optimization.AutoForwardDiff())
optprob_2 = Optimization.OptimizationProblem(optf_2, res_1.u)
res_2 = Optimization.solve(optprob_2, OptimizationOptimisers.Adam(0.002), maxiters=400, callback=callback) 

p_final = res_2.u 

# --- 6. 结果生成与详细评价 ---
println("\n" * "="^30)
println("      模型最终评价报告")
println("="^30)

# 1. 执行最终模拟
function calculate_nse(sim, obs)
    n_len = min(length(sim), length(obs))
    s, o = sim[1:n_len], obs[1:n_len]
    den = sum(abs2, o .- mean(o))
    return den < 1e-9 ? -999.0 : 1.0 - (sum(abs2, s .- o) / den)
end

p_phy_final = get_phys_params_mapped(p_final[1:12]) 
h_init_final = get_initial_h_vector(p_phy_final, raw_sm[:, 1])

# 使用最终参数和 h_init 进行最终求解
prob_final = remake(prob, u0=h_init_final, p=p_final) 
sol_final = solve(prob_final, Rodas5(), saveat=t_data, tstops=tstops_points, maxiters=200000)

# 2. 准备绘图与评价
plt = plot(layout=(6, 1), size=(900, 1400), link=:x, margin=8mm)
params_top_fin = [p_phy_final[1], p_phy_final[2], p_phy_final[3], p_phy_final[7], p_phy_final[8]]
params_sub_fin = [p_phy_final[4], p_phy_final[5], p_phy_final[6], p_phy_final[9], p_phy_final[8]]

all_nse = Float64[]

# *** 修改输出格式，仅输出 NSE ***
@printf("%-10s | %-10s\n", "深度(cm)", "NSE")
println("-"^22)

plot!(plt, t_data, raw_rain_cm, subplot=1, label="Rain", fill=true, color=:green, alpha=0.3, lw=0)
title!(plt, subplot=1, "Rainfall Input (Rows $START_ROW to $END_ROW)")

for (k, grid_idx) in enumerate(obs_indices)
    sim_h = sol_final[grid_idx, :]
    z_val = z_nodes[grid_idx]
    
    par = (z_val <= DEPTH_THRESHOLD) ? params_top_fin : params_sub_fin
    sim_theta = [vgm_robust_fixed(h, par)[1] for h in sim_h]
    obs_theta = raw_sm[k, :]
    
    nse_val = calculate_nse(sim_theta, obs_theta)
    push!(all_nse, nse_val)
    
    @printf("%-10.1f | %-10.4f\n", obs_depths[k], nse_val)
    
    plot!(plt, t_data, sim_theta, subplot=k+1, label="Sim", lw=2, color=:red, alpha=0.8)
    plot!(plt, t_data, obs_theta, subplot=k+1, label="Obs", lw=1.5, color=:blue)
    title!(plt, subplot=k+1, "Depth: $(obs_depths[k])cm | NSE: $(round(nse_val, digits=3))")
end

avg_nse = mean(all_nse)
println("-"^22)
@printf("全剖面平均 NSE: %.4f\n", avg_nse)
println("="^30)
savefig(plt, joinpath(OUTPUT_DIR, "$(START_ROW)~$(END_ROW)-4.png"))
println("完成！图片已保存至结果输出目录。")
