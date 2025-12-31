using DifferentialEquations
using Flux
using DiffEqFlux
using Interpolations
using Optimization
using OptimizationOptimisers
using OptimizationPolyalgorithms
using OptimizationOptimJL 
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
const OUTPUT_DIR = "结果输出/结果输出_Attention"
if !isdir(OUTPUT_DIR); mkdir(OUTPUT_DIR); end
const F = Float64
const DEPTH_THRESHOLD = 8.0 # 土层分界阈值m

# ========================================================
# 辅助函数
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

# 计算饱和水汽压差 (VPD) kPa
function calc_vpd(T_degC, RH_percent)
    es = 0.6108 * exp(17.27 * T_degC / (T_degC + 237.3))
    ea = es * (RH_percent / 100.0)
    return max(0.0, es - ea)
end

# ========================================================
# [物理参数映射] 
# ========================================================
function get_phys_params_mapped(p)
    # 使用 sigmoid 映射到物理范围
    log_alpha_top = -4.5 + 3.5 * sigmoid(p[1])   
    n_top         = 1.05 + 0.55 * sigmoid(p[2])   
    log_Ks_top    = log(0.5) + (log(100.0) - log(0.5)) * sigmoid(p[3]) 
    theta_s_top   = 0.30 + 0.25 * sigmoid(p[7]) 
    
    # Subsoil
    log_alpha_sub = log_alpha_top - (0.1 + 1.5 * sigmoid(p[4])) 
    n_sub         = 1.05 + (n_top - 1.05 ) * sigmoid(p[5]) 
    log_Ks_sub    = log_Ks_top - (0.5 + 3.0 * sigmoid(p[6])) 
    theta_s_sub   = 0.20 + (theta_s_top - 0.20) * sigmoid(p[9]) 
    
    # Common
    theta_r_val   = 0.001 + 0.08 * sigmoid(p[8]) 
    log_I_max_val = log(0.001) + (log(10.0) - log(0.001)) * sigmoid(p[10]) 
    alpha_et      = 0.05 + 0.45 * sigmoid(p[11]) 
    root_decay    = 20.0 + 80.0 * sigmoid(p[12]) 

    return (log_alpha_top, n_top, log_Ks_top, 
            log_alpha_sub, n_sub, log_Ks_sub,
            theta_s_top, theta_r_val, theta_s_sub, 
            log_I_max_val, alpha_et, root_decay)
end

# --- 1. 物理模型 (VGM) ---
function vgm_scalar(h, log_alpha, n, log_Ks, theta_s, theta_r)
    α = exp(log_alpha)
    Ks = exp(log_Ks)
    T = eltype(h); epsilon = T(1e-5)
    if h < -epsilon
        m = 1.0 - 1.0/n; abs_h = abs(h)
        denom = (1.0 + (α * abs_h)^n)
        Se = denom^(-m)
        theta = theta_r + (theta_s - theta_r) * Se
        num = (theta_s - theta_r) * m * n * α^n * (abs_h^(n-1))
        C = num / (denom^(m+1))
        K_rel = sqrt(Se) * (1.0 - (1.0 - Se^(1/m))^m)^2
        return theta, max(C, T(1e-4)), max(Ks * K_rel, T(1e-8))
    else
        return theta_s, T(1e-3), Ks
    end
end

function inverse_vgm_robust(theta_val, VGM_params)
    α = exp(VGM_params[1]); n = VGM_params[2]
    θs, θr = VGM_params[4], VGM_params[5]
    m = 1.0 - 1.0/n
    th_safe = max(θr + 1e-4, min(θs - 1e-4, theta_val))
    Se = (th_safe - θr) / (θs - θr)
    term = Se^(-1.0/m) - 1.0
    return term < 0 ? 0.0 : -((term^(1.0/n)) / α)
end

# --- 2. 神经网络 (升级为 6 个特征) ---
# 输入特征: [地表水头, 太阳辐射, VPD, 气温, 降雨状态, ΔT(地气温差)]
const N_FEATURES = 6

nn_model = Flux.Chain(
    Flux.Dense(N_FEATURES, 10, tanh),
    Flux.Dense(10, 1, sigmoid)
)
nn_model = Flux.f64(nn_model)
p_nn_init, re_nn = Flux.destructure(nn_model)

# --- 3. ODE 系统 ---
struct RichardsParams
    z_nodes::Vector{F}
    dz::Vector{F}
    rain_itp::AbstractInterpolation
    solar_itp::AbstractInterpolation 
    vpd_itp::AbstractInterpolation   
    temp_itp::AbstractInterpolation
    deltat_itp::AbstractInterpolation # [新增] 地气温差插值器
    obs_indices::Vector{Int}
end

function richards_ode_optimized!(dhdt, h, p, t, const_params::RichardsParams)
    # 1. 解包物理参数 (前12个)
    (log_α_top, n_top, log_Ks_top, log_α_sub, n_sub, log_Ks_sub,
     θs_top, θr, θs_sub, log_Imax, α_et, root_decay) = get_phys_params_mapped(p[1:12])
    
    # 2. 解包注意力参数 (接下来的 N_FEATURES 个)
    idx_attn_start = 13
    idx_attn_end = 13 + N_FEATURES - 1
    p_attn_raw = p[idx_attn_start:idx_attn_end]
    
    # Softmax 计算权重
    exps = exp.(p_attn_raw)
    attn_weights = exps ./ sum(exps)
    
    # 3. NN 参数
    p_nn_dynamic = p[idx_attn_end+1:end]

    I_max = exp(log_Imax)
    z = const_params.z_nodes; dz = const_params.dz; N = length(h)
    K = zeros(eltype(h), N); C = zeros(eltype(h), N)
    
    # 土壤属性计算
    for i in 1:N
        w = 1.0 - sigmoid((z[i] - DEPTH_THRESHOLD) / 5.0)
        curr_log_α  = w * log_α_top  + (1-w) * log_α_sub
        curr_n      = w * n_top      + (1-w) * n_sub
        curr_log_Ks = w * log_Ks_top + (1-w) * log_Ks_sub
        curr_θs     = w * θs_top     + (1-w) * θs_sub
        _, C[i], K[i] = vgm_scalar(max(-1e8, min(100.0, h[i])), curr_log_α, curr_n, curr_log_Ks, curr_θs, θr)
    end
    
    q = Vector{eltype(h)}(undef, N-1)
    for i in 1:N-1
        K_mid = sqrt(K[i] * K[i+1])
        q[i] = -K_mid * ((h[i+1] - h[i]) / dz[i] - 1.0)
    end
    
    # 4. [修改] 注意力感知与特征构造
    h_surf = h[1]
    
    # --- 特征工程 ---
    f_h     = tanh(log10(max(1.0, abs(h_surf))) / 5.0) # 1. 地表干湿 (0~1)
    f_solar = const_params.solar_itp(t) / 1000.0       # 2. 辐射 (0~1)
    f_vpd   = const_params.vpd_itp(t) / 3.0            # 3. VPD (0~1)
    f_temp  = const_params.temp_itp(t) / 35.0          # 4. 气温 (0~1)
    f_rain  = const_params.rain_itp(t) > 0.0 ? 1.0 : 0.0 # 5. 降雨状态
    
    # [新增] 6. 地气温差 (Delta T)
    # T_surf - T_air. 范围通常 -5 到 15。用 tanh 压缩到 -1~1 之间
    dt_raw = const_params.deltat_itp(t)
    f_dt   = tanh(dt_raw / 10.0) 
    
    raw_features = [f_h, f_solar, f_vpd, f_temp, f_rain, f_dt]
    
    # 注意力加权
    weighted_features = raw_features .* attn_weights
    
    # 预测 beta
    beta = re_nn(p_nn_dynamic)(weighted_features)[1]
    
    # 蒸散发计算
    E_pot_base = get_Ep_seasonal(t) 
    E_s_act = beta * (α_et * E_pot_base)         
    T_pot_scaled = beta * ((1.0 - α_et) * E_pot_base)   

    # 5. 根系与优先流
    Q_rain = const_params.rain_itp(t)
    Q_in_matrix_pot = min(Q_rain, I_max) 
    Q_bypass = max(0.0, Q_rain - Q_in_matrix_pot)
    idx_bypass_start = 6; idx_bypass_end = 10; lambda_bypass = 150.0 
    
    bypass_weights_sum = 0.0
    for j in idx_bypass_start:idx_bypass_end
        depth_rel = z[j] - z[idx_bypass_start]
        bypass_weights_sum += exp(-depth_rel / lambda_bypass)
    end

    total_root_weight = 0.0
    root_weights = zeros(eltype(h), N) 
    for i in 1:N
        root_weights[i] = exp(-z[i] / root_decay) + 0.05
        vol_factor = (i==1 ? dz[1] : (i==N ? dz[N-1] : 0.5*(dz[i-1]+dz[i])))
        total_root_weight += root_weights[i] * vol_factor
    end

    # 6. dhdt 计算
    for i in 1:N
        local_stress = sigmoid((h[i] + 15000.0) / 2000.0)
        sink_rate = T_pot_scaled * (root_weights[i] / total_root_weight) * local_stress
        
        if i == 1
            vol = 0.5 * dz[1]
            flux_in = (h[1] >= -10.0) ? min(Q_in_matrix_pot, K[1]) : Q_in_matrix_pot
            surf_stress = sigmoid((h[1] + 50000.0) / 5000.0)
            flux_out = q[1] + (E_s_act * 0.5 * surf_stress) 
        elseif i == N
            vol = 0.5 * dz[N-1]; flux_in = q[N-1]; flux_out = K[N] 
        else
            vol = 0.5 * (dz[i-1] + dz[i]); flux_in = q[i-1]; flux_out = q[i]
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

function get_Ep_seasonal(t)
    hour_of_day = t % 24
    diurnal = max(0.0, sin(pi * (hour_of_day - 6) / 12))
    season_factor = 0.5 
    return diurnal * season_factor
end

# --- 4. 数据加载 ---
println("=== 4. 数据加载与初始化 (方案B: 地气温差 ΔT) ===")
# 请根据实际情况修改路径
CSV_PATH = "g:/任鹏辉/孔老师任务/数据/USCRN-Filled_hourly_2024_sp54_Apr-Jun-1.csv" 
df = CSV.read(CSV_PATH, DataFrame; missingstring=["", "NA", "NaN", "null", "-9999"])

# [数据截取]
const START_ROW = 4370
const END_ROW = 6540
const N_SLICED_STEPS = END_ROW - START_ROW + 1 

if nrow(df) >= END_ROW
    println(">>> 已截取第 $START_ROW 行至第 $END_ROW 行 (共 $N_SLICED_STEPS 个数据点)。")
    df = df[START_ROW:END_ROW, :]
else
    error(">>> 错误：数据行数不足。")
end

# [处理基础数据]
raw_rain_mm = force_parse_float.(df[:, 15]) 
raw_rain_cm = Float64.(raw_rain_mm ./ 10.0)
N_total_steps = nrow(df)
t_data = Float64.(0:(N_total_steps-1)) 
rain_itp = LinearInterpolation(t_data, raw_rain_cm, extrapolation_bc=Interpolations.Flat())
rain_indices = findall(x -> x > 0.0, raw_rain_cm)
rain_times = t_data[rain_indices]
tstops_points = sort(unique(vcat(rain_times, rain_times .+ 0.5))) 

sm_cols = Matrix(df[:, 31:35])
raw_sm_vec = force_parse_float.(sm_cols)
raw_sm = raw_sm_vec' |> collect 
if mean(raw_sm) > 1.0; raw_sm .= raw_sm ./ 100.0; end

obs_depths = [5.0, 10.0, 20.0, 50.0, 100.0]
z_nodes = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 50.0, 75.0, 100.0]
obs_indices = [findfirst(x -> x==d, z_nodes) for d in obs_depths]

# [处理特征数据]
raw_solar = force_parse_float.(df[:, 16])
raw_temp  = force_parse_float.(df[:, 12])
raw_rh    = force_parse_float.(df[:, 29])
raw_stemp = force_parse_float.(df[:, 23]) # 地表辐射温度 SUR_TEMP

# 计算 VPD
raw_vpd = [calc_vpd(t, rh) for (t, rh) in zip(raw_temp, raw_rh)]

# [关键改进] 计算 Delta T = T_surf - T_air
raw_deltat = raw_stemp .- raw_temp
println(">>> ΔT 统计: Mean=$(mean(raw_deltat)), Max=$(maximum(raw_deltat)), Min=$(minimum(raw_deltat))")

# 建立插值器
solar_itp = LinearInterpolation(t_data, raw_solar, extrapolation_bc=Interpolations.Flat())
temp_itp  = LinearInterpolation(t_data, raw_temp, extrapolation_bc=Interpolations.Flat())
vpd_itp   = LinearInterpolation(t_data, raw_vpd, extrapolation_bc=Interpolations.Flat())
deltat_itp = LinearInterpolation(t_data, raw_deltat, extrapolation_bc=Interpolations.Flat())

# 更新常量结构体 (包含 deltat_itp)
const_params = RichardsParams(z_nodes, diff(z_nodes), rain_itp, solar_itp, vpd_itp, temp_itp, deltat_itp, obs_indices)

function get_initial_h_vector(p_mapped_tuple, obs_t0)
    (log_α_top, n_top, log_Ks_top, log_α_sub, n_sub, log_Ks_sub,
     θs_top, θr, θs_sub, _, _, _) = p_mapped_tuple
    
    params_top = [log_α_top, n_top, log_Ks_top, θs_top, θr]
    params_sub = [log_α_sub, n_sub, log_Ks_sub, θs_sub, θr]
    T = eltype(log_α_top)
    itp_init = LinearInterpolation(obs_depths, obs_t0, extrapolation_bc=Interpolations.Line())
    h0 = zeros(T, length(z_nodes)) 
    for i in 1:length(z_nodes)
        z_val = z_nodes[i]
        par = (z_val <= DEPTH_THRESHOLD) ? params_top : params_sub
        h0[i] = inverse_vgm_robust(itp_init(z_val), par)
    end
    return h0
end

# *** 初始参数 ***
p_phys_init = [
0.8,  1.2,  1.0,  # 1-3: Topsoil (略微降低 n 和 Ks，防止入渗太快导致峰值过冲)
0.5,  0.8,  2.8,  # 4-6: Subsoil (加大 p[6] 以显著降低深层导水率，阻止深层水分过快流失)
0.0, -1.0,  0.0,  # 7-9: θs, θr (保持或微调)
0.8,  2.5, -1.5   # 10-12: Imax降至0.8(压低峰值); αet增至2.5(加强表层蒸发); RootDecay降至-1.5(使根系变浅，消除深层波动)
]

# [新增] 注意力参数初始化 (6个特征)
p_attn_init = zeros(Float64, N_FEATURES) 
p_all_init = [p_phys_init; p_attn_init; p_nn_init] 
h_init_temp = zeros(length(z_nodes))
prob = ODEProblem((du, u, p, t) -> richards_ode_optimized!(du, u, p, t, const_params), 
                  h_init_temp, (t_data[1], t_data[end]), p_all_init)

# --- 5. 训练流程 ---
println("=== 5. 开始模型训练 (方案B) ===")

function get_smart_sampling_indices(steps, sparse_step=6)
    base_indices = 1:sparse_step:steps
    rain_event_indices = findall(x -> x > 0.0, raw_rain_cm[1:steps])
    combined_indices = sort(unique(vcat(base_indices, rain_event_indices)))
    if combined_indices[end] != steps; push!(combined_indices, steps); end
    return combined_indices
end

function loss_fn_flexible(p, target_sm, save_indices, tol_val=1e-2)
    t_save = t_data[save_indices]; t_end = t_save[end]
    p_mapped_tuple = get_phys_params_mapped(p[1:12])
    u0_dynamic = get_initial_h_vector(p_mapped_tuple, target_sm[:, 1])
    
    _prob = remake(prob, u0=u0_dynamic, tspan=(0.0, t_end), p=p)
    _sol = solve(_prob, TRBDF2(autodiff=false), saveat=t_save,
                 tstops=filter(x->x<=t_end, tstops_points),
                 abstol=tol_val, reltol=tol_val, maxiters=100000)
    
    if _sol.retcode != :Success; return 1e8; end
    
    (log_α_top, n_top, log_Ks_top, log_α_sub, n_sub, log_Ks_sub,
     θs_top, θr, θs_sub, _, _, _) = p_mapped_tuple
     
    loss = 0.0
    for (k, grid_idx) in enumerate(obs_indices)
        sim_h = _sol[grid_idx, :]
        z_val = z_nodes[grid_idx]
        curr_log_α  = (z_val <= DEPTH_THRESHOLD) ? log_α_top : log_α_sub
        curr_n      = (z_val <= DEPTH_THRESHOLD) ? n_top : n_sub
        curr_log_Ks = (z_val <= DEPTH_THRESHOLD) ? log_Ks_top : log_Ks_sub
        curr_θs     = (z_val <= DEPTH_THRESHOLD) ? θs_top : θs_sub
        
        sim_theta = [vgm_scalar(h, curr_log_α, curr_n, curr_log_Ks, curr_θs, θr)[1] for h in sim_h]
        loss += sum(abs2, sim_theta .- target_sm[k, save_indices]) * 10.0
    end
    return loss
end

callback = function (state, l)
    if l < 1e7; @printf("Loss: %.4f\r", l); else; print("Loss: High\r"); end
    return false
end

# 阶段 1: 快速热身
println("\n>>> 阶段 1/3: 快速热身...")
steps_1 = min(500, N_total_steps); indices_1 = get_smart_sampling_indices(steps_1, 2)
optf_1 = Optimization.OptimizationFunction((x,p)->loss_fn_flexible(x, raw_sm, indices_1, 1e-2), Optimization.AutoForwardDiff())
optprob_1 = Optimization.OptimizationProblem(optf_1, p_all_init)
res_1 = Optimization.solve(optprob_1, OptimizationOptimisers.Adam(0.05), maxiters=80, callback=callback)

# 阶段 2: 中期过渡
println("\n\n>>> 阶段 2/3: 中期过渡...")
steps_2 = min(1800, N_total_steps); indices_2 = get_smart_sampling_indices(steps_2, 4)
optf_2 = Optimization.OptimizationFunction((x,p)->loss_fn_flexible(x, raw_sm, indices_2, 1e-2), Optimization.AutoForwardDiff())
optprob_2 = Optimization.OptimizationProblem(optf_2, res_1.u)
res_2 = Optimization.solve(optprob_2, OptimizationOptimisers.Adam(0.01), maxiters=100, callback=callback)

# 阶段 3: 全局精细优化
println("\n\n>>> 阶段 3/3: 全局优化...")
steps_3 = min(1900, N_total_steps); indices_3 = get_smart_sampling_indices(steps_3, 6)
optf_3 = Optimization.OptimizationFunction((x,p)->loss_fn_flexible(x, raw_sm, indices_3, 1e-2), Optimization.AutoForwardDiff())
optprob_3 = Optimization.OptimizationProblem(optf_3, res_2.u)
res_3 = Optimization.solve(optprob_3, OptimizationOptimJL.LBFGS(), maxiters=150, callback=callback)

p_final = res_3.u 

# --- 6. 结果生成 ---
println("\n" * "="^30)
println("      模型最终评价报告 (方案B: ΔT)")
println("="^30)

idx_attn_start = 13
idx_attn_end = 13 + N_FEATURES - 1
final_attn_logits = p_final[idx_attn_start:idx_attn_end]
final_attn_weights = exp.(final_attn_logits) ./ sum(exp.(final_attn_logits))

println("最终特征注意力权重 (Feature Attention):")
feature_names = ["H_Surf (干湿)", "Solar (辐射)", "VPD (大气)", "Temp (气温)", "Rain (降雨)", "ΔT (植被/胁迫)"]
for (name, w) in zip(feature_names, final_attn_weights)
    @printf("  %-15s : %.4f\n", name, w)
end
println("-"^30)

function calculate_nse(sim, obs)
    n_len = min(length(sim), length(obs))
    s, o = sim[1:n_len], obs[1:n_len]
    den = sum(abs2, o .- mean(o))
    return den < 1e-9 ? -999.0 : 1.0 - (sum(abs2, s .- o) / den)
end

p_tuple_final = get_phys_params_mapped(p_final[1:12])
h_init_final = get_initial_h_vector(p_tuple_final, raw_sm[:, 1])

println("\n正在进行最终全时段高精度模拟...")
prob_final = remake(prob, u0=h_init_final, p=p_final, tspan=(t_data[1], t_data[end])) 
sol_final = solve(prob_final, TRBDF2(), saveat=t_data, tstops=tstops_points, abstol=1e-5, reltol=1e-5, maxiters=200000)

plt = plot(layout=(6, 1), size=(900, 1400), link=:x, margin=8mm)

(log_α_top, n_top, log_Ks_top, log_α_sub, n_sub, log_Ks_sub,
 θs_top, θr, θs_sub, _, _, _) = p_tuple_final

all_nse = Float64[]

@printf("%-10s | %-10s\n", "深度(cm)", "NSE(全段)")
println("-"^22)

plot!(plt, t_data, raw_rain_cm, subplot=1, label="Rain", fill=true, color=:green, alpha=0.3, lw=0)
vline!(plt, [t_data[steps_3]], subplot=1, color=:black, linestyle=:dash, label="Train End")
title!(plt, subplot=1, "Rainfall Input")

for (k, grid_idx) in enumerate(obs_indices)
    sim_h = sol_final[grid_idx, :]
    z_val = z_nodes[grid_idx]
    
    curr_log_α  = (z_val <= DEPTH_THRESHOLD) ? log_α_top : log_α_sub
    curr_n      = (z_val <= DEPTH_THRESHOLD) ? n_top : n_sub
    curr_log_Ks = (z_val <= DEPTH_THRESHOLD) ? log_Ks_top : log_Ks_sub
    curr_θs     = (z_val <= DEPTH_THRESHOLD) ? θs_top : θs_sub
    
    sim_theta = [vgm_scalar(h, curr_log_α, curr_n, curr_log_Ks, curr_θs, θr)[1] for h in sim_h]
    obs_theta = raw_sm[k, :]
    
    nse_val = calculate_nse(sim_theta, obs_theta)
    push!(all_nse, nse_val)
    
    @printf("%-10.1f | %-10.4f\n", obs_depths[k], nse_val)
    
    plot!(plt, t_data, sim_theta, subplot=k+1, label="Sim", lw=2, color=:red, alpha=0.8)
    plot!(plt, t_data, obs_theta, subplot=k+1, label="Obs", lw=1.5, color=:blue)
    vline!(plt, [t_data[steps_3]], subplot=k+1, color=:black, linestyle=:dash, label="")
    title!(plt, subplot=k+1, "Depth: $(obs_depths[k])cm | NSE: $(round(nse_val, digits=3))")
end

avg_nse = mean(all_nse)
println("-"^22)
@printf("全剖面平均 NSE: %.4f\n", avg_nse)
println("="^30)
savefig(plt, joinpath(OUTPUT_DIR, "$(START_ROW)~$(END_ROW)-A2.png"))
println("完成！结果已保存至 $OUTPUT_DIR")
