(* ::Package:: *)

(* ------------------------------------------------------------------ *)
(* eoms.wl                                                            *)
(*                                                                    *)
(* scalar Gauss-Bonnet (SGB) 理论在 (d+1) 维 FLRW 背景上的运动方程。     *)
(* 空间维数 d = dimx 全程保持符号，不在计算中代入 d = 3。                *)
(*                                                                    *)
(* 作用量：                                                            *)
(*   S = ∫ d^(d+1)x sqrt(-g) [ R/(2κ) - 1/2 (∂φ)^2 - V(φ)             *)
(*                              - ξ(φ)/16 L_GB + Λ ]                  *)
(*   L_GB = R^2 - 4 R_ab R^ab + R_abcd R^abcd                         *)
(*                                                                    *)
(* Λ 的符号与 papers/gauss-bonnet/renormalization-analysis.md 一致：    *)
(* 拉氏量里取 +Λ，对应 ρ_Λ = -Λ、p_Λ = +Λ。                             *)
(* 若要用通常的宇宙学常数约定（L ⊃ -Λ），把 Λ 换成 -Λ 即可。            *)
(*                                                                    *)
(* ★ 重要约定：本文件输出的所有方程都保留 φ 的空间梯度项，不做任何       *)
(*   "齐次背景" / "忽略梯度" 的近似。即                                    *)
(*       gradPhiSq = γ^ab ∂_aφ ∂_bφ        （物理量 = gradPhiSq/a^2）   *)
(*       lapPhi    = γ^ab ∇_a∇_bφ          （物理量 = lapPhi/a^2）      *)
(*   会原样出现在 eqPhi / eqHubble / eqAccel 及其导出量中。             *)
(*   若想要均匀背景方程，请自行令 gradPhiSq -> 0、lapPhi -> 0；         *)
(*   本文件不再提供忽略梯度的版本。                                     *)
(*                                                                    *)
(* 输出：eoms.output.wl（用 xTools`Misc` 的                            *)
(* WriteDefinitionsAsDelayedValueTo 写出，可直接 Get 回来）。           *)
(* 注意：读回结果前需要先加载 definitions.wl。                          *)
(* ------------------------------------------------------------------ *)

$here = If[$InputFileName =!= "", DirectoryName[$InputFileName], Directory[]];
Get[FileNameJoin[{$here, "definitions.wl"}]];
Needs["xTools`Misc`"];

(* 统一的后处理：去掉 IndexScope 包裹、规范化哑指标再化简 *)
cleanUp = ExpandScalarIndexScope /* ITensorReduce /* Simplify;

(* ================================================================== *)
(* 1. 作用量                                                          *)
(* ================================================================== *)

LGB = RiemMf[DI@a, DI@b, DI@c, DI@d] RiemMf[a, b, c, d] +
     (-4) RicMf[DI@a, DI@b] RicMf[a, b] +
     RsMf^2;

lagrangian = (RsMf/(2 κ) -
     1/2 cdMf[phi, DI@a] cdMf[phi, a] -
     potV[phi] -
     xi[phi]/16 LGB +
     Λ);

L = PutIndexScope[Sqrt[-detg] lagrangian, {}];

(* ================================================================== *)
(* 2. 变分：δS/δφ 与 δS/δg^{ab}                                      *)
(* ================================================================== *)

DefPerturbationOperator[pert];
DefTensorVariationOperator[var, VarCovDs -> {cdMf}, VarParamDs -> {pdT}];

L1 = pert[L, 1] // ExpandAllPerturbations[pert, mfInfo] // IndexScope //
     DeepContractMetric // ITensorReduce;

(* (d+1) 维协变运动方程；eomMetricTensor 的自由指标为 a, b *)
eomPhiTensor = cleanUp@(var[L1, pert[phi, 1]]/Sqrt[-detg]);
eomMetricTensor = cleanUp@(var[L1, pert[gMf][1, DI@a, DI@b]]/Sqrt[-detg]);

(* ================================================================== *)
(* 3. 在 FLRW 度规上取值（φ 仍可依赖空间坐标，梯度项未被丢弃）          *)
(* ================================================================== *)

eomPhiFrw = eomPhiTensor // NITensorReduceExtract[{}, frw] //
     DeepContractMetric // ITensorReduce // cleanUp;

eomMetFrw = eomMetricTensor // NITensorReduceExtract[{a, b}, frw] //
     DeepContractMetric // ITensorReduce;

eomTtFrw = cleanUp@eomMetFrw[[1, 1, 1]];
eomIjFrw = cleanUp@eomMetFrw[[2, 2, 1]];

(* 几何量，供校验 *)
rsOnFrw = cleanUp@(RsMf // NITensorReduceExtract[{}, frw] // DeepContractMetric);
gbOnFrw = cleanUp@(LovelockDensity[RiemMf, 2] //
     NITensorReduceExtract[{}, frw] // DeepContractMetric);

(* ================================================================== *)
(* 4. 运动方程（全部保留 ∇φ、∇²φ）                                     *)
(* ================================================================== *)

(* 用 H 表示：ȧ = aH, ä = a(Ḣ + H²)。这一步只替换 a，不碰 φ 的梯度。 *)
hubbleRules = {
    pdT[scaleA, t] :> scaleA hubbleH[t],
    pdT[scaleA, t, t] :> scaleA (pdT[hubbleH, t] + hubbleH[t]^2)
};

(* 梯度项的记号化：任何哑指标 i 上的 γ^ab ∂_aφ ∂_bφ -> gradPhiSq，
   γ^ab ∇_a∇_bφ -> lapPhi。只改写记号，不丢失任何项。 *)
gradientNotationRules = {
    cdMx[cdMx[phi, i_], DI[i_]] :> lapPhi,
    cdMx[phi, i_]*cdMx[phi, DI[i_]] :> gradPhiSq
};

(* --- φ 的运动方程 --- *)
(*   eqPhi = 0  <=>  φ̈ + dHφ̇ + V' + (GB 项) = ∇²φ/a²                *)
eqPhi = Simplify[(eomPhiFrw /. hubbleRules) //. gradientNotationRules];
eqPhiSolved = Simplify@First@Solve[eqPhi == 0, pdT[phi, t, t]];

(* --- tt 方程（Hubble 约束）---
   归一化：eqHubble = -2 κ a^4 E_tt
   tt 分量不含 ä，所以它仍是 H 的多项式（梯度项只进 H^0 和 H^2 系数） *)
eqHubble = Simplify[-2 scaleA^4 κ ((eomTtFrw /. hubbleRules) //.
       gradientNotationRules)];
eqHubblePolynomial = Collect[Expand[eqHubble], hubbleH[t], Simplify];
eqHubbleCoefs = CoefficientList[eqHubblePolynomial, hubbleH[t]];

(* --- ij 方程（加速方程）---
   E_ab 严格分解为
      E_ab = c1 γ_ab + c2 ∂_aφ ∂_bφ + c3 ∇_a∇_bφ + c4 ∇_b∇_aφ
   空间迹为 γ^{ab}E_ab = d c1 + c2 (∇φ)^2 + (c3 + c4) ∇²φ。
   归一化：eqAccel = -2 κ a^2 (γ^{ab}E_ab)/d *)
ijTraceRules = {
    gMx[DI[a], DI[b]] -> dimx,
    cdMx[phi, DI[a]]*cdMx[phi, DI[b]] -> gradPhiSq,
    cdMx[cdMx[phi, DI[a]], DI[b]] -> lapPhi,
    cdMx[cdMx[phi, DI[b]], DI[a]] -> lapPhi
};

eqAccel = Simplify[-2 scaleA^2 κ (Expand[eomIjFrw /. ijTraceRules] /.
       hubbleRules)/dimx] //. gradientNotationRules;
eqAccelSolved = Simplify@Solve[eqAccel == 0, pdT[hubbleH, t]];

(* ================================================================== *)
(* 5. 校验                                                            *)
(* ================================================================== *)

d3 = {dimx -> 3, k -> 0, κ -> 1};

(* 文章 (3.1)：3H² = 1/2 φ̇² + (∇φ)²/(2a²) + V - Λ
           - 1/2 H²(∇²φ ξ' + (∇φ)² ξ'')/a² + 3/2 H³ φ̇ ξ' *)
expectedD3Constraint = (3 hubbleH[t]^2 -
     pdT[phi, t]^2/2 - gradPhiSq/(2 scaleA^2) - potV[phi] + Λ +
     1/2 hubbleH[t]^2 (lapPhi Derivative[1][xi][phi] +
         gradPhiSq Derivative[2][xi][phi])/scaleA^2 -
     3/2 hubbleH[t]^3 pdT[phi, t] Derivative[1][xi][phi]);

(* d=3, k=0 的 φ 方程：φ̈ + 3Hφ̇ + V' + (3/2)H²(Ḣ+H²)ξ' = ∇²φ/a² *)
expectedD3Phi = (pdT[phi, t, t] + 3 hubbleH[t] pdT[phi, t] +
     Derivative[1][potV][phi] +
     3/2 hubbleH[t]^2 (pdT[hubbleH, t] + hubbleH[t]^2)
         Derivative[1][xi][phi] - lapPhi/scaleA^2);

(* d=3 的 ρ 与 p：ρ 含梯度能，p 含 -g²/6 *)
rhoD3 = pdT[phi, t]^2/2 + gradPhiSq/(2 scaleA^2) + potV[phi] - Λ;
pD3 = pdT[phi, t]^2/2 - gradPhiSq/(6 scaleA^2) - potV[phi] + Λ;

checks = <|
    "LGBMatchesLovelockDensity" -> Simplify[
        gbOnFrw - cleanUp@(LGB // NITensorReduceExtract[{}, frw] // DeepContractMetric)],
    "RicciScalar" -> Simplify[
        rsOnFrw - dimx ((dimx - 1) (k + pdT[scaleA, t]^2) +
            2 scaleA pdT[scaleA, t, t])/scaleA^2],
    "GaussBonnet" -> Simplify[
        gbOnFrw - dimx (dimx - 1) (dimx - 2) (k + pdT[scaleA, t]^2)
            ((dimx - 3) (k + pdT[scaleA, t]^2) +
            4 scaleA pdT[scaleA, t, t])/scaleA^4],
    "TtHasNoAAcceleration" -> FreeQ[eomTtFrw, pdT[scaleA, t, t]],

    (* 梯度项确实被保留 *)
    "PhiEomContainsGradients" -> !FreeQ[eqPhi, lapPhi],
    "ConstraintContainsGradients" -> !FreeQ[eqHubble, gradPhiSq | lapPhi],
    "AccelContainsGradients" -> !FreeQ[eqAccel, gradPhiSq | lapPhi],

    (* d=3 极限：含梯度的文章 (3.1) *)
    "D3ConstraintMatchesArticle" -> FreeQ[
        Simplify[(eqHubble /. d3)/expectedD3Constraint],
        hubbleH | phi | potV | xi | gradPhiSq | lapPhi],

    (* d=3 极限：φ 方程 *)
    "D3PhiMatchesArticle" -> FreeQ[
        Simplify[(eqPhi /. d3) + expectedD3Phi],
        hubbleH | phi | potV | xi | gradPhiSq | lapPhi],

    (* ξ=0：约束回到标准 FLRW 3H² = κρ（ρ 含梯度能） *)
    "D3ConstraintReducesToStandard" -> Simplify[
        (eqHubble /. xi -> (0 &) /. d3) - scaleA^4 (3 hubbleH[t]^2 - rhoD3)],

    (* ξ=0：迹方程回到 2Ḣ + 3H² = -p（p 含 -g²/6） *)
    "D3AccelReducesToStandard" -> Simplify[
        (eqAccel /. xi -> (0 &) /. d3) +
        scaleA^4 (2 pdT[hubbleH, t] + 3 hubbleH[t]^2 + pD3)],

    (* d=3 时纯 GB 部分消失（GB 在 4 维是拓扑项） *)
    "GaussBonnetTopologicalInD3" -> Simplify[
        Coefficient[eqHubblePolynomial /. d3, hubbleH[t], 4]]
|>;

(* ================================================================== *)
(* 6. 输出                                                            *)
(* ================================================================== *)

conventions = <|
    "Dimensions" -> "dimf = dimx + 1 (spatial dimension d = dimx is symbolic)",
    "Signature" -> "(-,+,+,+)",
    "Metric" -> "ds^2 = -dt^2 + a(t)^2 gamma_ab dx^a dx^b; gamma_ab is a constant-curvature slice with curvature k",
    "Action" -> "S = int d^(dimx+1)x sqrt(-g) [ R/(2kappa) - 1/2 (d phi)^2 - V(phi) - xi(phi)/16 LGB + Lambda ]",
    "GaussBonnet" -> "LGB = R^2 - 4 R_ab R^ab + R_abcd R^abcd",
    "LambdaSign" -> "L contains +Lambda, hence rho_Lambda = -Lambda and p_Lambda = +Lambda (matches renormalization-analysis.md)",
    "Kappa" -> "kappa = 8 pi G; the paper uses kappa = 1",
    "GradientTermsKept" -> "ALL equations keep the phi gradients; no homogeneous/background approximation is taken anywhere",
    "GradientSymbols" -> "gradPhiSq = gamma^ab d_a phi d_b phi, lapPhi = gamma^ab nabla_a nabla_b phi (divide by a^2 for physical quantities)",
    "Normalization" -> "eqHubble = -2 kappa a^4 E_tt; eqAccel = -2 kappa a^2 (gamma^ab E_ab)/d",
    "EnergyDensityConvention" -> "rho = 1/2 phidot^2 + gradPhiSq/(2a^2) + V - Lambda, p = 1/2 phidot^2 - gradPhiSq/(6a^2) - V + Lambda (d = 3)"
|>;

Print["---------------------------------------------------------------"];
Print["R on FLRW  : ", rsOnFrw];
Print["LGB on FLRW: ", gbOnFrw];
Print["---------------------------------------------------------------"];
Print["NOTE: every equation below keeps the phi gradients"];
Print["      gradPhiSq = gamma^ab d_a phi d_b phi"];
Print["      lapPhi    = gamma^ab nabla_a nabla_b phi"];
Print["---------------------------------------------------------------"];
Print["phi EOM:"];
Print["  ", eqPhi];
Print["phi'' solved:"];
Print["  ", eqPhiSolved];
Print["---------------------------------------------------------------"];
Print["Hubble constraint (polynomial in H):"];
Print["  ", eqHubblePolynomial];
Print["coefficients of H^0, H^1, ...:"];
Print["  ", eqHubbleCoefs];
Print["--- d = 3, k = 0, kappa = 1 ---"];
Print["  ", Simplify[eqHubblePolynomial /. d3]];
Print["---------------------------------------------------------------"];
Print["Acceleration equation (spatial trace of the ij equation):"];
Print["  ", eqAccel];
Print["Hdot solved:"];
Print["  ", eqAccelSolved];
Print["--- d = 3, k = 0, kappa = 1 ---"];
Print["  ", Simplify[eqAccel /. d3]];
Print["---------------------------------------------------------------"];
Print["checks:"];
Scan[Print["  ", #[[1]], " -> ", #[[2]]] &, List @@ Normal@checks];
Print["---------------------------------------------------------------"];

saveDefinitions[stream_, defs_List] := Scan[
    WriteString[stream, #[[1]] <> " := " <> ToString[InputForm[#[[2]]]] <> ";\n"] &,
    defs];

definitionsToSave = {
    {"conventions", conventions},
    {"gradientNotationRules", gradientNotationRules},
    {"rsOnFrw", rsOnFrw},
    {"gbOnFrw", gbOnFrw},
    {"eomPhiTensor", eomPhiTensor},
    {"eomMetricTensor", eomMetricTensor},
    {"eomPhiFrw", eomPhiFrw},
    {"eomTtFrw", eomTtFrw},
    {"eomIjFrw", eomIjFrw},
    {"eqPhi", eqPhi},
    {"eqPhiSolved", eqPhiSolved},
    {"eqHubble", eqHubble},
    {"eqHubblePolynomial", eqHubblePolynomial},
    {"eqHubbleCoefs", eqHubbleCoefs},
    {"eqAccel", eqAccel},
    {"eqAccelSolved", eqAccelSolved},
    {"checks", checks}
};

With[{outFile = FileNameJoin[{$here, "eoms.output.wl"}]},
    With[{stream = OpenWrite[outFile]},
        WriteString[stream,
            "(* ::Package:: *)\n\n" <>
            "(* eoms.output.wl -- generated by eoms.wl on " <> DateString[] <> " *)\n" <>
            "(* (d+1)-dimensional FLRW, dimf = dimx+1, signature (-,+,+,+). *)\n" <>
            "(* S = int d^(dimx+1)x sqrt(-g) [ R/(2kappa) - 1/2 (d phi)^2 - V(phi) - xi(phi)/16 LGB + Lambda ]. *)\n" <>
            "(* LGB = R^2 - 4 R_ab R^ab + R_abcd R^abcd. *)\n" <>
            "(* Lambda enters with a + sign in L, i.e. rho_Lambda = -Lambda. *)\n" <>
            "(*\n" <>
            "   ALL equations keep the phi gradients. No homogeneous/background\n" <>
            "   approximation is taken anywhere in this file.\n" <>
            "     gradPhiSq = gamma^ab d_a phi d_b phi   (physical: /a^2)\n" <>
            "     lapPhi    = gamma^ab nabla_a nabla_b phi (physical: /a^2)\n" <>
            "   Set gradPhiSq -> 0 and lapPhi -> 0 yourself if you want the\n" <>
            "   strictly homogeneous equations; they are NOT provided here.\n" <>
            " *)\n" <>
            "(* Load definitions.wl before using these symbols. *)\n\n" <>
            "(* Saved symbols:\n" <>
            "     conventions           -- conventions (association)\n" <>
            "     gradientNotationRules -- rules that map cdMx[...] to lapPhi / gradPhiSq\n" <>
            "                              (reverse them to substitute the cdMx form back)\n" <>
            "     rsOnFrw, gbOnFrw      -- R and LGB on FLRW\n" <>
            "     eomPhiTensor          -- covariant phi EOM in (d+1) dims (=0)\n" <>
            "     eomMetricTensor       -- covariant metric EOM in (d+1) dims (=0, free indices a,b)\n" <>
            "     eomPhiFrw, eomTtFrw, eomIjFrw -- the same evaluated on FLRW\n" <>
            "     eqPhi, eqPhiSolved    -- phi EOM / solved for phi'' (keeps lapPhi)\n" <>
            "     eqHubble, eqHubblePolynomial, eqHubbleCoefs -- Hubble constraint (keeps gradPhiSq, lapPhi)\n" <>
            "     eqAccel, eqAccelSolved -- acceleration equation / solved for Hdot (keeps gradPhiSq, lapPhi)\n" <>
            "     checks                -- verification results\n" <>
            "   All equations are implicit: the left-hand side vanishes on shell.\n" <>
            " *)\n\n"];
        saveDefinitions[stream, definitionsToSave];
        Close[stream];
        Print["written: ", outFile];
    ]
];
