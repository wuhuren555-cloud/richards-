using Base.Threads
using Optim
using DataInterpolations
using Statistics
using CSV
using DataFrames
using Dates
using Printf
using Plots
using DifferentialEquations
using Random
using StatsPlots  

Random.seed!(42)

# 文件路径配置 (请确保本地路径正确)
const FILE_SM   = joinpath("mechin learning", "code", "infiltration", "data", "sm", "除异-CHinaSM_2018-2019.csv")
const FILE_PRCP = joinpath("mechin learning", "code", "infiltration", "data", "rainfall", "prcp_CHM_PRE_V2_ChinaSM_2018-2019_sp2702.csv")
const OUTPUT_DIR = joinpath("mechin learning", "code", "infiltration", "results")
const OUTPUT_FILE = joinpath(OUTPUT_DIR, "追求高分-双层物理模型-ChinaSM_daily.csv") 

if !isdir(OUTPUT_DIR); mkpath(OUTPUT_DIR); end

const io_lock = ReentrantLock()
const plot_lock = ReentrantLock()

struct VGParams
    theta_r::Float64
    theta_s::Float64
    alpha::Float64
    n::Float64
    Ks::Float64
end

struct ProfileParams
    vg_nodes::NTuple{10, VGParams}  
    C_drain::Float64      
    f_macro_max::Float64  # 最大开裂时的截留上限
    split_idx::Float64    
    Z_macro::Float64      # 新增：裂隙能到达的最大深度边界
end

const dz = 100.0 
const nodes = collect(100.0:dz:1000.0) 
const N_nodes = length(nodes)

sigmoid(x) = 1.0 / (1.0 + exp(-x))

# ==============================================================================
# 1. 放宽参数边界 + 引入裂隙深度 Z_macro (12维参数)
# ==============================================================================
function get_mapped_params_2layer(p_latent)
    # A层 (表土层)
    Ks_A    = 10.0^(-1.0 + 3.0 * sigmoid(p_latent[1])) 
    alpha_A = 0.005 + 0.145 * sigmoid(p_latent[2])     
    n_A     = 1.1 + 1.4 * sigmoid(p_latent[3])         
    ths_A   = 0.35 + 0.25 * sigmoid(p_latent[4])       
    
    # B层 (底土层)
    Ks_B    = 10.0^(-4.0 + 5.0 * sigmoid(p_latent[5])) 
    alpha_B = 0.005 + 0.095 * sigmoid(p_latent[6])     
    n_B     = 1.1 + 0.9 * sigmoid(p_latent[7])         
    ths_B   = 0.30 + 0.20 * sigmoid(p_latent[8])       
    
    C_drain = 0.0 + 0.2 * sigmoid(p_latent[9])
    f_macro_max = 0.0 + 0.4 * sigmoid(p_latent[10]) # 允许更大裂隙截留
    split_idx = 1.5 + 7.0 * sigmoid(p_latent[11])
    
    # 🌟 物理革新：裂隙深度参数化 (10cm 到 100cm)
    Z_macro = 10.0 + 90.0 * sigmoid(p_latent[12])
    
    return[Ks_A, alpha_A, n_A, ths_A, Ks_B, alpha_B, n_B, ths_B, C_drain, f_macro_max, split_idx, Z_macro]
end

function get_hydro_props_bonan(theta, p::VGParams)
    th = clamp(theta, p.theta_r + 1e-6, p.theta_s - 1e-6)
    Se = (th - p.theta_r) / (p.theta_s - p.theta_r)
    m = 1.0 - 1.0/p.n
    K = (p.Ks * 10.0) * sqrt(Se) * (1.0 - (1.0 - Se^(1/m))^m)^2
    psi_mm = (1.0 / p.alpha * 10.0) * ((Se^(-1/m) - 1.0)^(1.0/p.n))
    
    h_cm = psi_mm / 10.0
    alpha = p.alpha; n_val = p.n
    if h_cm < 1e-5
        C = 1e-5
    else
        num = (p.theta_s - p.theta_r) * alpha * n_val * m * (alpha * h_cm)^(n_val - 1.0)
        den = (1.0 + (alpha * h_cm)^n_val)^(m + 1.0)
        C = max(num / den, 1e-6) 
    end
    D = K / (C / 10.0)
    return K, psi_mm, D
end

function richards_1d_ode!(du, u, p, t)
    p_profile, dz_val, q_in, S_arr, S_ET = p
    N = length(u)
    get_p_vg(i) = p_profile.vg_nodes[i]
    
    p_vg_1 = get_p_vg(1)
    th_1 = clamp(u[1], p_vg_1.theta_r + 1e-6, p_vg_1.theta_s - 1e-6)
    th_2 = clamp(u[2], get_p_vg(2).theta_r + 1e-6, get_p_vg(2).theta_s - 1e-6)
    K_lo_1, _, D_lo_1 = get_hydro_props_bonan(0.5*(th_1+th_2), p_vg_1)
    q_out_1 = -D_lo_1 * (th_2 - th_1)/dz_val + K_lo_1
    du[1] = (q_in - q_out_1)/dz_val + S_arr[1] - S_ET[1]
    
    q_in_i = q_out_1
    for i in 2:N-1
        p_vg_curr = get_p_vg(i); p_vg_next = get_p_vg(i+1)
        th_curr = clamp(u[i], p_vg_curr.theta_r + 1e-6, p_vg_curr.theta_s - 1e-6)
        th_next = clamp(u[i+1], p_vg_next.theta_r + 1e-6, p_vg_next.theta_s - 1e-6)
        
        K_lo, _, D_lo = get_hydro_props_bonan(0.5*(th_curr+th_next), p_vg_curr)
        q_out_i = -D_lo * (th_next - th_curr)/dz_val + K_lo
        du[i] = (q_in_i - q_out_i)/dz_val + S_arr[i] - S_ET[i]
        q_in_i = q_out_i
    end
    
    p_vg_N = get_p_vg(N)
    th_N = clamp(u[N], p_vg_N.theta_r + 1e-6, p_vg_N.theta_s - 1e-6)
    K_N, _, _ = get_hydro_props_bonan(th_N, p_vg_N)
    
    # 解决 80/100cm 水分流失挂不住的问题。水不到饱和边缘，几乎不流失。
    Se_N = clamp((th_N - p_vg_N.theta_r) / (p_vg_N.theta_s - p_vg_N.theta_r), 0.0, 1.0)
    q_out_N = K_N * exp(10.0 * (Se_N - 1.0)) * p_profile.C_drain 
    
    du[N] = (q_in_i - q_out_N)/dz_val + S_arr[N] - S_ET[N]
end

# ==============================================================================
# 2. 模拟运行器 (严格物理质量守恒重构)
# ==============================================================================
function run_advanced_simulation(u0, times_hours, precip_series, dates_series, p::ProfileParams)
    n_steps = length(times_hours)
    results = zeros(Float64, N_nodes, n_steps)
    results[:, 1] = u0
    
    out_runoff = zeros(n_steps); out_bypass = zeros(n_steps)
    out_infil  = zeros(n_steps); out_drain  = zeros(n_steps)
    
    theta_curr = copy(u0)
    p_top = p.vg_nodes[1]
    dummy_p = (p, dz, 0.0, zeros(N_nodes), zeros(N_nodes))
    base_prob = ODEProblem(richards_1d_ode!, theta_curr, (0.0, 1.0), dummy_p)
    
    root_weights =[exp(-3.0 * (i * dz) / 1000.0) for i in 1:N_nodes]
    root_weights ./= sum(root_weights)
    
    for t_idx in 1:(n_steps-1)
        this_dt_total = times_hours[t_idx+1] - times_hours[t_idx]
        if this_dt_total <= 0.0; this_dt_total = 24.0; end
        
        raw_rain = precip_series[t_idx+1]
        daily_rain = raw_rain > 1.5 ? (raw_rain - 1.5) * 0.95 : 0.0
        
        if daily_rain <= 0.0; rain_duration = 0.0
        elseif daily_rain < 20.0; rain_duration = 6.0 
        elseif daily_rain < 50.0; rain_duration = 4.0  
        else; rain_duration = 2.0; end
        
        rain_duration = min(rain_duration, this_dt_total)
        dry_duration  = this_dt_total - rain_duration
        
        phases =[]
        if rain_duration > 0.0; push!(phases, (rain_duration, daily_rain / rain_duration)); end
        if dry_duration > 0.0; push!(phases, (dry_duration, 0.0)); end
        
        cum_runoff_step = 0.0; cum_bypass_step = 0.0
        cum_infil_step  = 0.0; cum_drain_step  = 0.0
        
        for (phase_dt, rain_rate) in phases
            Se_top = clamp((theta_curr[1] - p_top.theta_r) / (p_top.theta_s - p_top.theta_r), 0.0, 1.0)
            K_sat_top_mm_h = p_top.Ks * 10.0
            
            # 达西表面入渗能力
            _, psi_top_mm, _ = get_hydro_props_bonan(theta_curr[1], p_top)
            hydraulic_gradient_top = (abs(psi_top_mm) / 50.0) + 1.0
            max_infil_cap = K_sat_top_mm_h * hydraulic_gradient_top
            
            actual_surface_infil = min(rain_rate, max_infil_cap)
            excess_rain = rain_rate - actual_surface_infil
            
            #  表面干裂开合机制 (天干裂缝大，天湿裂缝合)
            crack_openness_surf = (1.0 - Se_top)^2
            effective_f_macro = p.f_macro_max * crack_openness_surf
            
            bypass_rate = excess_rain * effective_f_macro
            runoff_rate = excess_rain - bypass_rate
            
            S_array = zeros(Float64, N_nodes)
            if bypass_rate > 0.0
                q_macro_current = bypass_rate 
                
                #  物理革新：漏斗式向下演进机制 (Leaky Pipe Routing)
                for i in 2:N_nodes
                    pv = p.vg_nodes[i]
                    depth_cm = i * 10.0
                    Se_i = clamp((theta_curr[i] - pv.theta_r) / (pv.theta_s - pv.theta_r), 0.0, 1.0)
                    
                    # 1. 管径受土层含水率胀缩影响 + 深度衰减 (到达Z_macro时管径趋于0)
                    crack_openness_i = (1.0 - Se_i)^2
                    depth_taper = exp(- (depth_cm / p.Z_macro)^2)
                    pipe_fraction = depth_taper * crack_openness_i
                    
                    # 2. 能继续往下一层流的水量
                    Q_pass = q_macro_current * pipe_fraction
                    
                    # 3. 因管径收缩而被强制“憋”出来挤入本层基质的水量
                    Q_exit = q_macro_current - Q_pass
                    
                    # 4. 基质主动侧壁毛细抽吸水量 (依赖该层导水率和干湿度)
                    suction_power = (pv.Ks * 10.0) * 0.02 
                    suction_fraction = (1.0 - exp(-suction_power)) * (1.0 - Se_i)
                    Q_suction = Q_pass * suction_fraction
                    
                    # 扣除被抽走的水后，剩余向下的真实水流
                    Q_pass_final = Q_pass - Q_suction
                    
                    # 5. 总计尝试灌入该层基质的优先流水量
                    Q_try_matrix = Q_exit + Q_suction
                    
                    # 6. 该层物理能容纳的绝对极限 (防止数值爆炸，消灭诡异尖峰)
                    max_volume_mmh = max(0.0, (pv.theta_s - theta_curr[i])) * dz / phase_dt * 0.95
                    Q_actual_matrix = min(Q_try_matrix, max_volume_mmh)
                    
                    # 7. 🌟 绝对质量守恒：被挤出但基质装不下的水，发生“拥堵倒灌”，回到地表转为径流！
                    Q_rejected = Q_try_matrix - Q_actual_matrix
                    
                    S_array[i] = Q_actual_matrix / dz
                    cum_bypass_step += Q_actual_matrix * phase_dt
                    cum_runoff_step += Q_rejected * phase_dt # 拥堵溢出
                    
                    q_macro_current = Q_pass_final
                    if q_macro_current < 1e-5; break; end
                end
                
                # 穿透整根管子没有被吸收完的水，直接深层补给
                if q_macro_current > 0.0
                    cum_drain_step += q_macro_current * phase_dt
                end
            end
            
            PET_rate_mm_h = max(0.5, 1.0 + 3.5 * sin((Dates.month(dates_series[t_idx+1]) - 3) * π / 6)) / 24.0
            if rain_rate > 0.0; PET_rate_mm_h *= 0.2; end 
            
            S_ET_array = zeros(Float64, N_nodes)
            for i in 1:N_nodes
                pv = p.vg_nodes[i]
                S_ET_array[i] = (PET_rate_mm_h * root_weights[i] * clamp((theta_curr[i]-pv.theta_r)/(pv.theta_s-pv.theta_r), 0.0, 1.0)) / dz
            end
            
            prob_step = remake(base_prob; u0 = theta_curr, p = (p, dz, actual_surface_infil, S_array, S_ET_array), tspan = (0.0, phase_dt))
            sol = solve(prob_step, Rosenbrock23(), reltol=1e-3, abstol=1e-3)
            theta_next = sol.u[end]
            
            rejected_infil = 0.0
            for i in 1:N_nodes
                pv = p.vg_nodes[i]
                if theta_next[i] > pv.theta_s
                    rejected_infil += (theta_next[i] - pv.theta_s) * dz
                    theta_curr[i] = pv.theta_s
                elseif theta_next[i] < pv.theta_r + 1e-6
                    theta_curr[i] = pv.theta_r + 1e-6
                else
                    theta_curr[i] = theta_next[i]
                end
            end
            
            cum_runoff_step += runoff_rate * phase_dt + rejected_infil
            cum_infil_step  += actual_surface_infil * phase_dt - rejected_infil
            
            # 使用带阈值的指数型排水物理底界
            K_bot_end = get_hydro_props_bonan(theta_curr[N_nodes], p.vg_nodes[N_nodes])[1]
            Se_N_end = clamp((theta_curr[N_nodes] - p.vg_nodes[N_nodes].theta_r) / (p.vg_nodes[N_nodes].theta_s - p.vg_nodes[N_nodes].theta_r), 0.0, 1.0)
            cum_drain_step  += K_bot_end * exp(10.0 * (Se_N_end - 1.0)) * p.C_drain * phase_dt
        end 
        
        results[:, t_idx+1]  = theta_curr
        out_runoff[t_idx+1]  = cum_runoff_step
        out_bypass[t_idx+1]  = cum_bypass_step
        out_infil[t_idx+1]   = cum_infil_step
        out_drain[t_idx+1]   = cum_drain_step
    end
    return results, out_runoff, out_bypass, out_infil, out_drain
end

function calc_kge(sim, obs)
    std_sim = max(std(sim), 1e-6); std_obs = max(std(obs), 1e-6)
    r = cor(sim, obs)
    if isnan(r); r = 0.0; end 
    α = std_sim / std_obs
    β = mean(sim) / max(abs(mean(obs)), 1e-6)
    return max(1.0 - sqrt((r - 1.0)^2 + (α - 1.0)^2 + (β - 1.0)^2), -9.99)
end

function calc_nse(sim, obs)
    d = sum((obs .- mean(obs)).^2)
    return d == 0.0 ? NaN : 1.0 - sum((sim .- obs).^2) / d
end

# ==============================================================================
# 3. 数据处理与方差加权优化
# ==============================================================================
function safe_parse_float(x); ismissing(x) && return missing; x isa Number && return Float64(x); s = strip(string(x)); (s == "" || s == "NA" || s == "NaN") && return missing; v = tryparse(Float64, s); return isnothing(v) ? missing : v; end
function safe_parse_date(x); x isa Date && return x; x isa DateTime && return Date(x); s = replace(strip(string(x)), "/" => "-"); length(s) >= 10 && return Date(s[1:10]); length(s) == 8 && return Date(s, "yyyymmdd"); return Date(s); end

function process_site_data(site_id, df_site::DataFrame)
    cols_sm =[:depth_10, :depth_20, :depth_30, :depth_40, :depth_50, :depth_60, :depth_80, :depth_100]
    cols_needed = [[:time, :prec]; cols_sm]
    
    missing_cols = setdiff(cols_needed, propertynames(df_site))
    if !isempty(missing_cols); return (false, "缺少必须列", nothing, nothing, nothing) end
    
    df_clean = select(df_site, cols_needed)
    for col in[cols_sm; :prec]; df_clean[!, col] = safe_parse_float.(df_clean[!, col]); end
    dropmissing!(df_clean)
    
    filter!(r -> r.depth_10 >= 0.0 && r.depth_10 <= 100.0 && r.depth_100 >= 0.0 && r.depth_100 <= 100.0, df_clean)
    df_clean.prec =[x < 0 ? 0.0 : x for x in df_clean.prec]
    if nrow(df_clean) < 10; return (false, "有效数据过少", nothing, nothing, nothing) end
    for col in cols_sm; df_clean[!, col] = df_clean[!, col] ./ 100.0; end
    try; df_clean.Date_Parsed = safe_parse_date.(df_clean.time); catch e; return (false, "日期解析失败", nothing, nothing, nothing) end
    sort!(df_clean, :Date_Parsed)
    
    t_start = df_clean.Date_Parsed[1]
    df_clean.Time_Hours =[Dates.value(d - t_start) * 24.0 for d in df_clean.Date_Parsed]
    unique!(df_clean, :Time_Hours)
    if nrow(df_clean) < 10; return (false, "时间去重后数据过少", nothing, nothing, nothing) end
    
    obs_cols =[:depth_10, :depth_20, :depth_30, :depth_40, :depth_50, :depth_60, :depth_80, :depth_100]
    depths_obs =[100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0]
    
    obs_mins =[max(0.01, quantile(df_clean[!, col], 0.01) - 0.03) for col in obs_cols]
    itp_min = LinearInterpolation(obs_mins, depths_obs)
    thr_nodes =[itp_min(nodes[i]) for i in 1:N_nodes]
    
    vals_t0 = [df_clean[1, col] for col in obs_cols]
    itp_prof = LinearInterpolation(vals_t0, depths_obs)
    obs_all = [df_clean[!, col] for col in obs_cols]
    sim_idxs =[1, 2, 3, 4, 5, 6, 8, 10]

    warmup_days = 7
    start_idx = findfirst(x -> x >= warmup_days * 24.0, df_clean.Time_Hours)
    if isnothing(start_idx); start_idx = 1; end
    
    obs_stds =[max(std(obs[start_idx:end]), 1e-3) for obs in obs_all]
    layer_weights = obs_stds ./ sum(obs_stds) 

    function loss_function(p_latent)
        p_vec = get_mapped_params_2layer(p_latent)
        split_idx = p_vec[11]
        Z_macro = p_vec[12]
        
        vg_list = ntuple(10) do i
            w_A = 1.0 / (1.0 + exp(-5.0 * (split_idx - i)))
            w_B = 1.0 - w_A
            
            Ks_node = 10.0^(w_A * log10(p_vec[1]) + w_B * log10(p_vec[5])) 
            alpha_node = w_A * p_vec[2] + w_B * p_vec[6]
            n_node     = w_A * p_vec[3] + w_B * p_vec[7]
            ths_node   = max(thr_nodes[i] + 0.05, w_A * p_vec[4] + w_B * p_vec[8]) 
            
            VGParams(thr_nodes[i], ths_node, alpha_node, n_node, Ks_node)
        end
        p_profile = ProfileParams(vg_list, p_vec[9], p_vec[10], split_idx, Z_macro)
        
        u0_dynamic = zeros(N_nodes)
        for i in 1:N_nodes
            u0_dynamic[i] = clamp(itp_prof(nodes[i]), vg_list[i].theta_r + 1e-4, vg_list[i].theta_s - 1e-4)
        end
        
        sim_res, _, _, _, _ = run_advanced_simulation(u0_dynamic, df_clean.Time_Hours, df_clean.prec, df_clean.Date_Parsed, p_profile)
        
        loss = 0.0
        for (idx, obs_layer) in enumerate(obs_all)
            sim_slice = sim_res[sim_idxs[idx], start_idx:end]
            obs_slice = obs_layer[start_idx:end]
            kge_val = calc_kge(sim_slice, obs_slice)
            loss += layer_weights[idx] * (1.0 - kge_val)
        end
               
        # 轻量惩罚，将主动权交给物理守恒方程
        loss += 0.05 * sum((p_latent ./ 3.0).^4) 
        return loss
    end

    # 🌟 12 维参数多起点寻优
    opt_options_fast = Optim.Options(time_limit=15.0, iterations=500, show_trace=false)
    best_loss = Inf; best_p_init = zeros(12)
    
    for _ in 1:5
        p_rand = randn(12) .* 1.5 
        res_tmp = optimize(loss_function, p_rand, NelderMead(), opt_options_fast)
        if Optim.minimum(res_tmp) < best_loss
            best_loss = Optim.minimum(res_tmp)
            best_p_init = Optim.minimizer(res_tmp)
        end
    end
    
    opt_options_full = Optim.Options(time_limit=60.0, iterations=3000, show_trace=false)
    opt_res_final = optimize(loss_function, best_p_init, NelderMead(), opt_options_full)
    
    best_p_latent = Optim.minimizer(opt_res_final)
    best_p = get_mapped_params_2layer(best_p_latent)
    
    split_idx_opt = best_p[11]
    Z_macro_opt   = best_p[12]
    best_vg_list = ntuple(10) do i
        w_A = 1.0 / (1.0 + exp(-5.0 * (split_idx_opt - i)))
        w_B = 1.0 - w_A
        Ks_node = 10.0^(w_A * log10(best_p[1]) + w_B * log10(best_p[5]))
        alpha_node = w_A * best_p[2] + w_B * best_p[6]
        n_node     = w_A * best_p[3] + w_B * best_p[7]
        ths_node   = max(thr_nodes[i] + 0.05, w_A * best_p[4] + w_B * best_p[8])
        VGParams(thr_nodes[i], ths_node, alpha_node, n_node, Ks_node)
    end
    best_profile = ProfileParams(best_vg_list, best_p[9], best_p[10], split_idx_opt, Z_macro_opt)
    
    u0_final = zeros(N_nodes)
    for i in 1:N_nodes
        u0_final[i] = clamp(itp_prof(nodes[i]), best_vg_list[i].theta_r + 1e-4, best_vg_list[i].theta_s - 1e-4)
    end
    
    final_sim, runoff, bypass, infil, drain = run_advanced_simulation(u0_final, df_clean.Time_Hours, df_clean.prec, df_clean.Date_Parsed, best_profile)
    
    avg_kge_final = 0.0
    for (i, obs_layer) in zip(sim_idxs, obs_all)
        sim_slice = final_sim[i, start_idx:end]
        obs_slice = obs_layer[start_idx:end]
        avg_kge_final += calc_kge(sim_slice, obs_slice)
    end
    avg_kge_final /= 8.0

    N_rows = nrow(df_clean)
    res_storage = [sum(final_sim[:, i]) * dz for i in 1:N_rows]
    
    res_df = DataFrame(
        Site_ID = fill(string(site_id), N_rows), Date = df_clean.Date_Parsed, Precipitation_mm = df_clean.prec,
        Profile_Storage_mm = round.(res_storage, digits=2), Total_Infiltration_mm = round.(infil .+ bypass, digits=2), 
        Surface_Runoff_mm = round.(runoff, digits=2), Cumulative_Drainage_mm = round.(drain, digits=2),
        Opt_Ks_Topsoil = fill(round(best_p[1], digits=2), N_rows), Opt_alpha_Top = fill(round(best_p[2], digits=4), N_rows),
        Opt_n_Top = fill(round(best_p[3], digits=3), N_rows), Opt_ths_Top = fill(round(best_p[4], digits=3), N_rows),
        Opt_Ks_Subsoil = fill(round(best_p[5], digits=2), N_rows), Opt_alpha_Sub = fill(round(best_p[6], digits=4), N_rows),
        Opt_n_Sub = fill(round(best_p[7], digits=3), N_rows), Opt_ths_Sub = fill(round(best_p[8], digits=3), N_rows),
        Opt_Z_Split_cm = fill(round(best_p[11] * 10.0, digits=1), N_rows), 
        Opt_Z_Macro_cm = fill(round(best_p[12], digits=1), N_rows), # 🌟 新增保存列
        Opt_Cdrain = fill(round(best_p[9], digits=3), N_rows),
        Opt_MacroFrac = fill(round(best_p[10], digits=3), N_rows), Average_KGE_Score = fill(round(avg_kge_final, digits=4), N_rows)
    )
    
    lock(plot_lock) do 
        try
            max_rain_idx = argmax(df_clean.prec)
            start_plot_idx = max(1, max_rain_idx - 7)
            end_plot_idx = min(N_rows, start_plot_idx + 29) 
            if end_plot_idx - start_plot_idx < 29 && start_plot_idx > 1; start_plot_idx = max(1, end_plot_idx - 29); end
            
            dates_month = string.(df_clean.Date_Parsed[start_plot_idx:end_plot_idx]) 
            rain_month  = df_clean.prec[start_plot_idx:end_plot_idx]
            infil_month = (infil .+ bypass)[start_plot_idx:end_plot_idx]
            
            tick_idx = 1:4:length(dates_month)
            my_xticks = (tick_idx, dates_month[tick_idx])
            y_max_rain = maximum(rain_month) > 0.1 ? maximum(rain_month) * 1.2 : 10.0
            
            p_rain = groupedbar(dates_month,[rain_month infil_month], labels=["Rainfall (mm)" "Total Infiltration (mm)"], 
                 color=[:blue :orange], bar_position=:dodge, bar_width=0.7, linecolor=:match,
                 ylabel="Flux (mm)", ylims=(0.0, y_max_rain), title="Site $site_id (Split@$(round(best_p[11]*10,digits=1))cm | MacroDepth:$(round(best_p[12],digits=1))cm)",
                 legend=:topright, framestyle=:box, xrotation=45, xticks=my_xticks)
                         
            obs_cols =[:depth_10, :depth_20, :depth_30, :depth_40, :depth_50, :depth_60, :depth_80, :depth_100]
            depth_labels =["10cm", "20cm", "30cm", "40cm", "50cm", "60cm", "80cm", "100cm"]
            
            sm_plots =[] 
            for i in 1:8
                col = obs_cols[i]; s_idx = sim_idxs[i]; d_lab = depth_labels[i]
                
                obs_full = df_clean[!, col]; sim_full = final_sim[s_idx, :]
                kge_val = calc_kge(sim_full[start_idx:end], obs_full[start_idx:end])
                nse_val = calc_nse(sim_full[start_idx:end], obs_full[start_idx:end])
                
                kge_str = kge_val <= -4.9 ? "N/A" : @sprintf("%.2f", kge_val)
                nse_str = isnan(nse_val)    ? "N/A" : @sprintf("%.2f", nse_val)
                
                obs_val = df_clean[start_plot_idx:end_plot_idx, col]; sim_val = final_sim[s_idx, start_plot_idx:end_plot_idx]
                ymin = min(minimum(obs_val), minimum(sim_val)); ymax = max(maximum(obs_val), maximum(sim_val))
                if ymax - ymin < 0.02; ymin = max(0.0, ymin - 0.05); ymax = ymin + 0.10; else; ymin = max(0.0, ymin - 0.02); ymax = ymax + 0.02; end
                
                p_sm = plot(dates_month, obs_val, label="Obs", color=:black, lw=2, markershape=:circle)
                plot!(p_sm, dates_month, sim_val, label="Sim", color=:red, lw=2, ls=:dash)
                plot!(p_sm, title="$d_lab | KGE: $kge_str | NSE: $nse_str", titlefontsize=9); ylabel!(p_sm, "θ")
                plot!(p_sm, xrotation=45, framestyle=:box, legend=:topright, ylims=(ymin, ymax), legendfontsize=7, tickfontsize=8, guidefontsize=9, xticks=my_xticks)
                push!(sm_plots, p_sm)
            end
            
            l = @layout[a{0.2h}; grid(4, 2)]
            fig = plot(p_rain, sm_plots..., layout=l, size=(1200, 1600), margin=6Plots.mm)
            savefig(fig, joinpath(OUTPUT_DIR, "Site_$(site_id)_1Month_HighFit.png"))
        catch e
            println("⚠️ 站点 $site_id 绘图失败: $e")
        end
    end

    return (true, res_df, best_p, avg_kge_final)
end

function main()
    println("正在读取输入文件...")
    df_sm = CSV.read(FILE_SM, DataFrame; missingstring=["-9999", "NaN", "NA", ""])
    df_prcp = CSV.read(FILE_PRCP, DataFrame; missingstring=["-9999", "NaN", "NA", ""])
    df_sm.site = string.(df_sm.site); df_prcp.site = string.(df_prcp.site)
    
    println("合并数据中...")
    df_merged = innerjoin(df_sm, df_prcp, on=[:site, :time])
    
    target_sites = unique(df_merged.site)[1:min(10, length(unique(df_merged.site)))] 
    println("启动多线程，测试 $(length(target_sites)) 个站点...")
    
    if isfile(OUTPUT_FILE); rm(OUTPUT_FILE); end
    state = Ref(true)
    
    Threads.@threads for i in 1:length(target_sites)
        site_id = target_sites[i]
        thread_id = Threads.threadid()
        
        try
            df_subset = filter(row -> row.site == site_id, df_merged)
            success, result, best_p, score = process_site_data(site_id, df_subset)
            
            if success
                lock(io_lock) do
                    is_first = state[]
                    CSV.write(OUTPUT_FILE, result; append=!is_first, writeheader=is_first)
                    if is_first; state[] = false; end
                    println("✅[T$thread_id] 站点 $site_id | 均KGE=$(round(score, digits=3)) | 裂隙深=$(round(best_p[12],digits=1))cm | 孔隙度[Top:$(round(best_p[4],digits=2)), Sub:$(round(best_p[8],digits=2))]")
                end
            end
        catch e
            println("❌[T$thread_id] 站点 $site_id 异常: $e")
        end
    end
    println("完毕！结果位于: $OUTPUT_FILE")
end

main()
